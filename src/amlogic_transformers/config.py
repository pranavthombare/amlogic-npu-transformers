from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    id: str
    revision: str = "main"
    layer_index: int = 0
    onnx_opset: int = 17
    torch_dtype: str = "float32"


@dataclass
class TargetConfig:
    board: str = "VIM3"
    os: str = "khadas-ubuntu-bsp"
    sdk_env: str = "AML_NPU_SDK"


@dataclass
class ArtifactConfig:
    root: Path
    manifest_name: str = "artifact_manifest.json"


@dataclass
class GraphConfig:
    name: str
    sequence_length: int
    batch_size: int = 1


@dataclass
class AcuityConfig:
    platform: str = "onnx"
    quantized_dtype: str = "dynamic_fixed_point"
    qtype: str = "int8"
    print_level: int = 1
    extra_args: list[str] = field(default_factory=list)


@dataclass
class RuntimeConfig:
    adapter_library: str = "libamlogic_transformers_adapter.so"
    atol: float = 0.08
    rtol: float = 0.08
    min_cosine_similarity: float = 0.98


@dataclass
class PipelineConfig:
    model: ModelConfig
    target: TargetConfig
    artifacts: ArtifactConfig
    graphs: list[GraphConfig]
    acuity: AcuityConfig
    runtime: RuntimeConfig

    @property
    def manifest_path(self) -> Path:
        return self.artifacts.root / self.artifacts.manifest_name


def _mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    model = ModelConfig(**_mapping(raw.get("model"), "model"))
    target = TargetConfig(**_mapping(raw.get("target", {}), "target"))
    artifacts_raw = _mapping(raw.get("artifacts"), "artifacts")
    artifacts = ArtifactConfig(root=Path(artifacts_raw["root"]), manifest_name=artifacts_raw.get("manifest_name", "artifact_manifest.json"))
    graphs = [GraphConfig(**_mapping(item, "graphs[]")) for item in raw.get("graphs", [])]
    if not graphs:
        raise ValueError("at least one graph must be configured")
    acuity = AcuityConfig(**_mapping(raw.get("acuity", {}), "acuity"))
    runtime = RuntimeConfig(**_mapping(raw.get("runtime", {}), "runtime"))
    return PipelineConfig(model=model, target=target, artifacts=artifacts, graphs=graphs, acuity=acuity, runtime=runtime)

