from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gemma_npu.config import load_config
from gemma_npu.convert_acuity import convert_graphs
from gemma_npu.manifest import ArtifactManifest, GraphArtifact, TensorInfo, load_manifest, write_manifest


class ConfigManifestTests(unittest.TestCase):
    def test_load_default_config(self) -> None:
        config = load_config("configs/gemma3_270m_block.yaml")
        self.assertEqual(config.model.id, "google/gemma-3-270m-it")
        self.assertEqual(config.target.board, "VIM3")
        self.assertEqual([graph.name for graph in config.graphs], ["prefill", "decode"])

    def test_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = ArtifactManifest(
                model_id="google/gemma-3-270m-it",
                model_revision="main",
                layer_index=0,
                board="VIM3",
                graphs=[
                    GraphArtifact(
                        name="prefill",
                        sequence_length=64,
                        batch_size=1,
                        onnx_path="prefill/prefill.onnx",
                        reference_inputs=[
                            TensorInfo(
                                name="hidden_states",
                                path="prefill/reference/hidden_states.npy",
                                dtype="float32",
                                shape=[1, 64, 640],
                            )
                        ],
                        reference_outputs=[
                            TensorInfo(
                                name="hidden_states_out",
                                path="prefill/reference/hidden_states_out.npy",
                                dtype="float32",
                                shape=[1, 64, 640],
                            )
                        ],
                    )
                ],
            )
            path = tmp_path / "artifact_manifest.json"
            write_manifest(path, manifest)
            loaded = load_manifest(path)
            self.assertEqual(loaded.model_id, manifest.model_id)
            self.assertEqual(loaded.graphs[0].reference_inputs[0].name, "hidden_states")

    def test_acuity_dry_run_updates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_config = json.loads(
                json.dumps(
                    {
                        "model": {
                            "id": "google/gemma-3-270m-it",
                            "revision": "main",
                            "layer_index": 0,
                            "onnx_opset": 17,
                            "torch_dtype": "float32",
                        },
                        "target": {"board": "VIM3", "os": "khadas-ubuntu-bsp", "sdk_env": "AML_NPU_SDK"},
                        "artifacts": {"root": str(tmp_path / "artifacts"), "manifest_name": "artifact_manifest.json"},
                        "graphs": [{"name": "prefill", "sequence_length": 64, "batch_size": 1}],
                        "acuity": {
                            "platform": "onnx",
                            "quantized_dtype": "dynamic_fixed_point",
                            "qtype": "int8",
                            "print_level": 1,
                            "extra_args": [],
                        },
                        "runtime": {
                            "adapter_library": "libgemma_aml_adapter.so",
                            "atol": 0.08,
                            "rtol": 0.08,
                            "min_cosine_similarity": 0.98,
                        },
                    }
                )
            )
            config_path = tmp_path / "config.yaml"
            import yaml

            config_path.write_text(yaml.safe_dump(project_config))
            config = load_config(config_path)
            (config.artifacts.root / "prefill" / "reference").mkdir(parents=True)
            (config.artifacts.root / "prefill" / "reference" / "hidden_states.npy").write_bytes(b"fixture")
            (config.artifacts.root / "prefill" / "reference" / "hidden_states_out.npy").write_bytes(b"fixture")

            manifest = ArtifactManifest(
                model_id=config.model.id,
                model_revision=config.model.revision,
                layer_index=0,
                board="VIM3",
                graphs=[
                    GraphArtifact(
                        name="prefill",
                        sequence_length=64,
                        batch_size=1,
                        onnx_path="prefill/prefill.onnx",
                        reference_inputs=[
                            TensorInfo(
                                name="hidden_states",
                                path="prefill/reference/hidden_states.npy",
                                dtype="float32",
                                shape=[1, 1, 4],
                            )
                        ],
                        reference_outputs=[
                            TensorInfo(
                                name="hidden_states_out",
                                path="prefill/reference/hidden_states_out.npy",
                                dtype="float32",
                                shape=[1, 1, 4],
                            )
                        ],
                    )
                ],
            )
            write_manifest(config.manifest_path, manifest)

            sdk_root = tmp_path / "sdk"
            convert_bin = sdk_root / "acuity-toolkit" / "python" / "convert"
            convert_bin.parent.mkdir(parents=True)
            convert_bin.write_text("#!/bin/sh\nexit 0\n")

            updated = convert_graphs(config, sdk_root=sdk_root, dry_run=True)
            self.assertTrue(updated.graphs[0].conversion_command)
            self.assertIn("--platform", updated.graphs[0].conversion_command)


if __name__ == "__main__":
    unittest.main()
