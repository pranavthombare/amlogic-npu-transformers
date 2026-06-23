from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install serving dependencies: pip install -e '.[serve]'") from exc

from gemma_npu.manifest import ArtifactManifest, load_manifest


MODEL_ID = "google/gemma-3-270m-it"
DEFAULT_MANIFEST = Path("artifacts/gemma3_270m/block0/artifact_manifest.json")


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    max_tokens: int = Field(default=64, ge=1, le=128)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False


class HealthResponse(BaseModel):
    ok: bool
    model: str
    manifest_path: str
    manifest_loaded: bool
    ready_graphs: list[str]
    missing: list[str]


def _manifest_path() -> Path:
    return Path(os.environ.get("GEMMA_NPU_MANIFEST", str(DEFAULT_MANIFEST)))


def _load_manifest_or_none() -> ArtifactManifest | None:
    path = _manifest_path()
    if not path.exists():
        return None
    return load_manifest(path)


def _readiness() -> HealthResponse:
    path = _manifest_path()
    manifest = _load_manifest_or_none()
    missing: list[str] = []
    ready_graphs: list[str] = []

    if manifest is None:
        missing.append(f"manifest:{path}")
    else:
        for graph in manifest.graphs:
            graph_missing: list[str] = []
            if not graph.nb_path:
                graph_missing.append(f"{graph.name}:nb_path")
            if not graph.adapter_library:
                graph_missing.append(f"{graph.name}:adapter_library")
            if graph_missing:
                missing.extend(graph_missing)
            else:
                ready_graphs.append(graph.name)

    return HealthResponse(
        ok=manifest is not None and not missing,
        model=MODEL_ID,
        manifest_path=str(path),
        manifest_loaded=manifest is not None,
        ready_graphs=ready_graphs,
        missing=missing,
    )


app = FastAPI(title="VIM3 Gemma NPU", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return _readiness()


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "google",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.model != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not supported in v1")

    readiness = _readiness()
    if not readiness.ok:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "NPU artifacts are not ready",
                "missing": readiness.missing,
                "next_step": "run export_gemma_block.py and convert_acuity.py, then provide an adapter library",
            },
        )

    raise HTTPException(
        status_code=501,
        detail=(
            "full token generation is intentionally disabled until the one-block "
            "NPU harness passes numerical reference checks"
        ),
    )


def openai_error_response(message: str, status: int = 500) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "vim3_gemma_npu_error",
            "code": status,
            "timestamp": int(time.time()),
        }
    }


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install serving dependencies: pip install -e '.[serve]'") from exc

    host = os.environ.get("GEMMA_NPU_HOST", "0.0.0.0")
    port = int(os.environ.get("GEMMA_NPU_PORT", "8080"))
    uvicorn.run("gemma_npu.server:app", host=host, port=port)
