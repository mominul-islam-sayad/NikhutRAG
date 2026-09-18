"""FastAPI backend, as committed in the FYDP-I paper's software stack (p.17).

    uvicorn app.api:app --reload
    curl -s localhost:8000/detect -H "content-type: application/json" \
         -d '{"question":"...","context":"...","answer":"..."}'

The model is loaded once at startup, not per request, so /detect latency is
comparable to the benchmark numbers in outputs/metrics.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app._loader import Bundle, analyze, find_checkpoints, load_bundle  # noqa: E402
from src import config as cfg  # noqa: E402

_state: dict[str, Bundle] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["bundle"] = load_bundle()
    yield
    _state.clear()


app = FastAPI(
    title="Bangla RAG Hallucination Detector",
    description="Token-level hallucination detection for RAG, running locally on a small encoder.",
    version="0.1.0",
    lifespan=lifespan,
)


class DetectRequest(BaseModel):
    question: str = Field("", description="The user question")
    context: str = Field(..., description="Retrieved context treated as ground truth")
    answer: str = Field(..., min_length=1, description="LLM answer to check")
    threshold: float = Field(cfg.SPAN_THRESHOLD, ge=0.0, le=1.0)
    with_baseline: bool = Field(True, description="Also run the lexical baseline")


class Span(BaseModel):
    start: int
    end: int
    text: str


class DetectResponse(BaseModel):
    hallucinated: bool
    score: float
    words: list[str]
    word_probs: list[float]
    word_labels: list[int]
    spans: list[Span]
    latency_ms: float
    run: str
    untrained: bool
    baseline: dict | None = None


@app.get("/health")
def health() -> dict:
    b = _state.get("bundle")
    return {
        "status": "ok" if b else "loading",
        "run": getattr(b, "name", None),
        "backbone": getattr(b, "backbone", None),
        "is_lora": getattr(b, "is_lora", None),
        "untrained": getattr(b, "name", None) == "UNTRAINED",
        "checkpoints_available": [p.name for p in find_checkpoints()],
    }


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> dict:
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return analyze(
        bundle,
        question=req.question,
        context=req.context,
        answer=req.answer,
        threshold=req.threshold,
        with_baseline=req.with_baseline,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)
