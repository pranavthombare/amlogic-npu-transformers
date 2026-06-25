from __future__ import annotations

import os
import argparse
import subprocess
from pathlib import Path

from amlogic_transformers.config import PipelineConfig, load_config
from amlogic_transformers.manifest import ArtifactManifest, GraphArtifact, load_manifest, write_manifest


def _sdk_root(config: PipelineConfig, sdk_root: Path | None) -> Path:
    if sdk_root:
        return sdk_root
    value = os.environ.get(config.target.sdk_env)
    if not value:
        raise RuntimeError(f"{config.target.sdk_env} is not set")
    return Path(value)


def _convert_bin(sdk_root: Path) -> Path:
    path = sdk_root / "acuity-toolkit" / "python" / "convert"
    if not path.exists():
        raise RuntimeError(f"Acuity convert tool not found: {path}")
    return path


def _graph_root(config: PipelineConfig, graph: GraphArtifact) -> Path:
    return config.artifacts.root / graph.name


def _abs_artifact(config: PipelineConfig, relative_path: str) -> Path:
    return config.artifacts.root / relative_path


def _write_calibration_list(config: PipelineConfig, graph: GraphArtifact) -> Path:
    calibration_dir = _graph_root(config, graph) / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = calibration_dir / "dataset0.txt"
    input_paths = [_abs_artifact(config, item.path).resolve() for item in graph.reference_inputs]
    if not input_paths:
        raise RuntimeError(f"graph {graph.name} has no reference inputs")
    dataset_path.write_text(" ".join(str(path) for path in input_paths) + "\n")
    return dataset_path


def _model_name(config: PipelineConfig, graph: GraphArtifact) -> str:
    model_slug = config.model.id.split("/")[-1].replace("-", "_")
    return f"{model_slug}_layer{config.model.layer_index}_{graph.name}"


def _conversion_command(
    config: PipelineConfig,
    graph: GraphArtifact,
    convert_bin: Path,
    dataset_path: Path,
) -> list[str]:
    onnx_path = _abs_artifact(config, graph.onnx_path).resolve()
    outputs = " ".join(item.name for item in graph.reference_outputs)
    inputs = " ".join(item.name for item in graph.reference_inputs)

    command = [
        str(convert_bin),
        "--model-name",
        _model_name(config, graph),
        "--platform",
        config.acuity.platform,
        "--model",
        str(onnx_path),
        "--inputs",
        inputs,
        "--outputs",
        outputs,
        "--quantized-dtype",
        config.acuity.quantized_dtype,
        "--source-files",
        str(dataset_path.resolve()),
        "--kboard",
        config.target.board,
        "--print-level",
        str(config.acuity.print_level),
    ]
    if config.acuity.qtype:
        command.extend(["--qtype", config.acuity.qtype])
    command.extend(config.acuity.extra_args)
    return command


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def convert_graphs(
    config: PipelineConfig,
    sdk_root: Path | None = None,
    dry_run: bool = False,
) -> ArtifactManifest:
    manifest = load_manifest(config.manifest_path)
    sdk = _sdk_root(config, sdk_root)
    convert_bin = _convert_bin(sdk)
    convert_cwd = convert_bin.parent

    updated_graphs: list[GraphArtifact] = []
    for graph in manifest.graphs:
        dataset_path = _write_calibration_list(config, graph)
        command = _conversion_command(config, graph, convert_bin, dataset_path)
        output_dir = convert_cwd / "outputs" / _model_name(config, graph)

        if not dry_run:
            subprocess.run(command, cwd=convert_cwd, check=True)

        graph.acuity_output_dir = str(output_dir)
        graph.conversion_command = command
        nb_path = _find_first(output_dir, "*.nb") if output_dir.exists() else None
        lib_path = _find_first(output_dir, "libnn_*.so") if output_dir.exists() else None
        graph.nb_path = str(nb_path) if nb_path else None
        graph.adapter_library = str(lib_path) if lib_path else graph.adapter_library
        updated_graphs.append(graph)

    manifest.graphs = updated_graphs
    write_manifest(config.manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert exported ONNX graphs with Amlogic Acuity")
    parser.add_argument("--config", "-c", required=True, type=Path, help="Pipeline YAML config")
    parser.add_argument("--sdk-root", type=Path, default=None, help="Amlogic SDK root; defaults to AML_NPU_SDK")
    parser.add_argument("--dry-run", action="store_true", help="Write commands to manifest without running Acuity")
    args = parser.parse_args()

    pipeline_config = load_config(args.config)
    manifest = convert_graphs(pipeline_config, sdk_root=args.sdk_root, dry_run=args.dry_run)
    print(f"updated {pipeline_config.manifest_path}")
    for graph in manifest.graphs:
        print(f"{graph.name}:")
        print("  " + " ".join(graph.conversion_command))


if __name__ == "__main__":
    main()
