from __future__ import annotations

import inspect
import os
import argparse
from pathlib import Path
from typing import Any

from amlogic_transformers.config import PipelineConfig, load_config
from amlogic_transformers.manifest import ArtifactManifest, GraphArtifact, TensorInfo, write_manifest


def _import_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("install export dependencies: pip install -e '.[export]'") from exc


def _import_transformers() -> Any:
    try:
        import transformers

        return transformers
    except ImportError as exc:
        raise RuntimeError("install export dependencies: pip install -e '.[export]'") from exc


def _torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {name}") from exc


def _model_backbone(model: Any) -> Any:
    for attr in ("model", "language_model", "transformer"):
        value = getattr(model, attr, None)
        if value is not None and hasattr(value, "layers"):
            return value
    raise RuntimeError("could not find model backbone with a .layers attribute")


def _hidden_size(model: Any, backbone: Any) -> int:
    value = getattr(model.config, "hidden_size", None)
    if value:
        return int(value)
    embed_tokens = getattr(backbone, "embed_tokens", None)
    if embed_tokens is not None and hasattr(embed_tokens, "weight"):
        return int(embed_tokens.weight.shape[1])
    raise RuntimeError("could not infer hidden size")


def _causal_mask(torch: Any, batch_size: int, sequence_length: int, dtype: Any) -> Any:
    mask = torch.zeros((sequence_length, sequence_length), dtype=dtype)
    if sequence_length > 1:
        upper = torch.triu(torch.ones_like(mask), diagonal=1).bool()
        mask = mask.masked_fill(upper, -10000.0)
    return mask.reshape(1, 1, sequence_length, sequence_length).repeat(batch_size, 1, 1, 1)


def _call_rotary(rotary_emb: Any, hidden_states: Any, position_ids: Any) -> Any:
    try:
        return rotary_emb(hidden_states, position_ids)
    except TypeError:
        return rotary_emb(hidden_states, position_ids=position_ids)


class DecoderBlockWrapper:
    def __init__(self, torch: Any, block: Any, rotary_emb: Any | None):
        self.torch = torch
        self.block = block
        self.rotary_emb = rotary_emb
        self.signature = inspect.signature(block.forward)

    def module(self) -> Any:
        torch = self.torch
        outer = self

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = outer.block

            def forward(self, hidden_states, attention_mask, position_ids, cache_position):
                kwargs: dict[str, Any] = {}
                params = outer.signature.parameters
                if "attention_mask" in params:
                    kwargs["attention_mask"] = attention_mask
                if "position_ids" in params:
                    kwargs["position_ids"] = position_ids
                if "cache_position" in params:
                    kwargs["cache_position"] = cache_position
                if "use_cache" in params:
                    kwargs["use_cache"] = False
                if "output_attentions" in params:
                    kwargs["output_attentions"] = False
                if "position_embeddings" in params:
                    if outer.rotary_emb is None:
                        raise RuntimeError("decoder block requires position_embeddings but no rotary_emb was found")
                    kwargs["position_embeddings"] = _call_rotary(outer.rotary_emb, hidden_states, position_ids)

                output = self.block(hidden_states, **kwargs)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                return output

        return Module()


def _onnx_input_names(onnx_path: Path) -> set[str]:
    try:
        import onnx
    except ImportError:
        return set()
    model = onnx.load(str(onnx_path))
    initializers = {item.name for item in model.graph.initializer}
    return {item.name for item in model.graph.input if item.name not in initializers}


def _tensor_info(name: str, path: Path, tensor: Any, root: Path) -> TensorInfo:
    return TensorInfo(
        name=name,
        path=str(path.resolve().relative_to(root.resolve())),
        dtype=str(tensor.detach().cpu().numpy().dtype),
        shape=list(tensor.shape),
    )


def _save_tensor(path: Path, tensor: Any) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("install export dependencies: pip install -e '.[export]'") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.detach().cpu().numpy())


def export_graphs(config: PipelineConfig, token: str | None = None) -> ArtifactManifest:
    torch = _import_torch()
    transformers = _import_transformers()
    dtype = _torch_dtype(torch, config.model.torch_dtype)
    torch.manual_seed(0)

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required because google/gemma-3-270m-it is gated")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        token=token,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    backbone = _model_backbone(model)
    layers = getattr(backbone, "layers")
    block = layers[config.model.layer_index].eval()
    rotary_emb = getattr(backbone, "rotary_emb", None)
    hidden_size = _hidden_size(model, backbone)

    root = config.artifacts.root
    root.mkdir(parents=True, exist_ok=True)
    graph_artifacts: list[GraphArtifact] = []

    wrapper = DecoderBlockWrapper(torch, block, rotary_emb).module().eval()

    for graph in config.graphs:
        graph_dir = root / graph.name
        graph_dir.mkdir(parents=True, exist_ok=True)

        hidden_states = torch.randn(
            graph.batch_size,
            graph.sequence_length,
            hidden_size,
            dtype=dtype,
        )
        attention_mask = _causal_mask(torch, graph.batch_size, graph.sequence_length, dtype)
        position_ids = torch.arange(graph.sequence_length, dtype=torch.long).reshape(1, graph.sequence_length)
        position_ids = position_ids.repeat(graph.batch_size, 1)
        cache_position = torch.arange(graph.sequence_length, dtype=torch.long)

        tensors = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }

        with torch.no_grad():
            output = wrapper(hidden_states, attention_mask, position_ids, cache_position)

        onnx_path = graph_dir / f"{graph.name}.onnx"
        torch.onnx.export(
            wrapper,
            (hidden_states, attention_mask, position_ids, cache_position),
            str(onnx_path),
            input_names=list(tensors.keys()),
            output_names=["hidden_states_out"],
            opset_version=config.model.onnx_opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )

        present_inputs = _onnx_input_names(onnx_path) or set(tensors.keys())
        reference_inputs: list[TensorInfo] = []
        for name, tensor in tensors.items():
            if name not in present_inputs:
                continue
            tensor_path = graph_dir / "reference" / f"{name}.npy"
            _save_tensor(tensor_path, tensor)
            reference_inputs.append(_tensor_info(name, tensor_path, tensor, root))

        output_path = graph_dir / "reference" / "hidden_states_out.npy"
        _save_tensor(output_path, output)
        reference_outputs = [_tensor_info("hidden_states_out", output_path, output, root)]

        graph_artifacts.append(
            GraphArtifact(
                name=graph.name,
                sequence_length=graph.sequence_length,
                batch_size=graph.batch_size,
                onnx_path=str(onnx_path.resolve().relative_to(root.resolve())),
                reference_inputs=reference_inputs,
                reference_outputs=reference_outputs,
            )
        )

    manifest = ArtifactManifest(
        model_id=config.model.id,
        model_revision=config.model.revision,
        layer_index=config.model.layer_index,
        board=config.target.board,
        graphs=graph_artifacts,
        metadata={"hidden_size": hidden_size, "torch_dtype": config.model.torch_dtype},
    )
    write_manifest(config.manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one Gemma decoder block to fixed-shape ONNX")
    parser.add_argument("--config", "-c", required=True, type=Path, help="Pipeline YAML config")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token; defaults to HF_TOKEN")
    args = parser.parse_args()

    pipeline_config = load_config(args.config)
    manifest = export_graphs(pipeline_config, token=args.hf_token)
    print(f"wrote {pipeline_config.manifest_path}")
    for graph in manifest.graphs:
        print(f"exported {graph.name}: {graph.onnx_path}")


if __name__ == "__main__":
    main()
