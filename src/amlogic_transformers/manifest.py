from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TensorInfo:
    name: str
    path: str
    dtype: str
    shape: list[int]


@dataclass
class GraphArtifact:
    name: str
    sequence_length: int
    batch_size: int
    onnx_path: str
    reference_inputs: list[TensorInfo] = field(default_factory=list)
    reference_outputs: list[TensorInfo] = field(default_factory=list)
    acuity_output_dir: str | None = None
    nb_path: str | None = None
    adapter_library: str | None = None
    conversion_command: list[str] = field(default_factory=list)


@dataclass
class ArtifactManifest:
    model_id: str
    model_revision: str
    layer_index: int
    board: str
    graphs: list[GraphArtifact]
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


def _tensor_from_dict(raw: dict[str, Any]) -> TensorInfo:
    return TensorInfo(
        name=raw["name"],
        path=raw["path"],
        dtype=raw["dtype"],
        shape=[int(item) for item in raw["shape"]],
    )


def _graph_from_dict(raw: dict[str, Any]) -> GraphArtifact:
    return GraphArtifact(
        name=raw["name"],
        sequence_length=int(raw["sequence_length"]),
        batch_size=int(raw["batch_size"]),
        onnx_path=raw["onnx_path"],
        reference_inputs=[_tensor_from_dict(item) for item in raw.get("reference_inputs", [])],
        reference_outputs=[_tensor_from_dict(item) for item in raw.get("reference_outputs", [])],
        acuity_output_dir=raw.get("acuity_output_dir"),
        nb_path=raw.get("nb_path"),
        adapter_library=raw.get("adapter_library"),
        conversion_command=list(raw.get("conversion_command", [])),
    )


def _manifest_from_dict(raw: dict[str, Any]) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=int(raw.get("schema_version", 1)),
        created_at=raw.get("created_at") or datetime.now(timezone.utc).isoformat(),
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        layer_index=int(raw["layer_index"]),
        board=raw["board"],
        graphs=[_graph_from_dict(item) for item in raw.get("graphs", [])],
        metadata=dict(raw.get("metadata", {})),
    )


def load_manifest(path: str | Path) -> ArtifactManifest:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("artifact manifest must contain a JSON object")
    return _manifest_from_dict(raw)


def write_manifest(path: str | Path, manifest: ArtifactManifest) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def relpath(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))

