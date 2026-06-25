from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict
from pathlib import Path

from amlogic_transformers.manifest import ArtifactManifest, GraphArtifact, TensorInfo, write_manifest


def _shape_literal(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(item) for item in shape) + ")"


def _write_npy_f32(path: Path, shape: tuple[int, ...], values: list[float]) -> None:
    element_count = 1
    for dim in shape:
        element_count *= dim
    if len(values) != element_count:
        raise ValueError(f"{path} expects {element_count} values, got {len(values)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + _shape_literal(shape) + ", }"
    header_len_without_padding = 10 + len(header) + 1
    padding = (16 - (header_len_without_padding % 16)) % 16
    encoded_header = (header + (" " * padding) + "\n").encode("ascii")

    with path.open("wb") as output:
        output.write(b"\x93NUMPY")
        output.write(bytes([1, 0]))
        output.write(struct.pack("<H", len(encoded_header)))
        output.write(encoded_header)
        output.write(struct.pack("<" + "f" * len(values), *values))


def make_fixture(output: Path, adapter: str) -> Path:
    root = output.resolve()
    graph_name = "prefill"
    graph_dir = root / graph_name
    reference_dir = graph_dir / "reference"
    values = [0.25, -0.5, 1.25, 2.0]
    shape = (1, 4)

    _write_npy_f32(reference_dir / "hidden_states.npy", shape, values)
    _write_npy_f32(reference_dir / "hidden_states_out.npy", shape, values)
    (graph_dir / "dummy.nb").write_bytes(b"local reference adapter fixture\n")

    manifest = ArtifactManifest(
        model_id="google/gemma-3-270m-it",
        model_revision="fixture",
        layer_index=0,
        board="VIM3",
        graphs=[
            GraphArtifact(
                name=graph_name,
                sequence_length=4,
                batch_size=1,
                onnx_path=f"{graph_name}/{graph_name}.onnx",
                reference_inputs=[
                    TensorInfo(
                        name="hidden_states",
                        path=f"{graph_name}/reference/hidden_states.npy",
                        dtype="float32",
                        shape=list(shape),
                    )
                ],
                reference_outputs=[
                    TensorInfo(
                        name="hidden_states_out",
                        path=f"{graph_name}/reference/hidden_states_out.npy",
                        dtype="float32",
                        shape=list(shape),
                    )
                ],
                nb_path=f"{graph_name}/dummy.nb",
                adapter_library=adapter,
            )
        ],
        metadata={"fixture": True},
    )
    manifest_path = root / "artifact_manifest.json"
    write_manifest(manifest_path, manifest)

    summary_path = root / "fixture_summary.json"
    summary_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny local runtime fixture")
    parser.add_argument("--output", type=Path, required=True, help="Fixture output directory")
    parser.add_argument(
        "--adapter",
        default="../libamlogic_transformers_reference_adapter.so",
        help="Adapter path stored in manifest",
    )
    args = parser.parse_args()

    manifest_path = make_fixture(args.output, args.adapter)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
