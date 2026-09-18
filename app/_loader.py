"""Checkpoint discovery and inference, shared by the Streamlit UI and the API.

Kept out of src/ deliberately: this is deployment glue, not part of the
training pipeline. Both entry points load through here so the UI and the API
can never disagree about what the model said.

Handles both checkpoint kinds transparently -- a full fine-tune saves model
weights, a LoRA run saves an adapter that needs its base model loaded first.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src import config as cfg  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src.baselines import LexicalOverlapBaseline  # noqa: E402
from src.model import build_model, predict_one  # noqa: E402

DEFAULT_MAX_LENGTH = 256


@dataclass
class Bundle:
    model: object
    tokenizer: object
    name: str
    backbone: str
    max_length: int
    is_lora: bool
    device: str = "cpu"
    metrics: dict = field(default_factory=dict)


def find_checkpoints() -> list[Path]:
    root = cfg.CHECKPOINT_DIR
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and any(p.iterdir()))


def _metrics_for(run_name: str) -> dict:
    path = cfg.METRICS_DIR / f"{run_name}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def load_bundle(checkpoint: str | Path | None = None, device: str = "cpu") -> Bundle:
    """Load a saved run, or fall back to the untrained backbone.

    An untrained fallback is deliberate: the UI should start and say plainly
    that predictions are meaningless, rather than crash before the first run
    exists.
    """
    if checkpoint is None:
        found = find_checkpoints()
        checkpoint = found[0] if found else None

    if checkpoint is None:
        backbone = cfg.BACKBONE
        model = build_model(backbone)
        tok = AutoTokenizer.from_pretrained(backbone)
        return Bundle(
            model=model.eval().to(device),
            tokenizer=tok,
            name="UNTRAINED",
            backbone=backbone,
            max_length=DEFAULT_MAX_LENGTH,
            is_lora=False,
            device=device,
        )

    path = Path(checkpoint)
    run_name = path.name
    metrics = _metrics_for(run_name)
    backbone = metrics.get("backbone", cfg.BACKBONE)
    max_length = int(metrics.get("max_length") or DEFAULT_MAX_LENGTH)
    is_lora = (path / "adapter_config.json").exists()

    if is_lora:
        from peft import PeftModel

        base = build_model(backbone)
        model = PeftModel.from_pretrained(base, path)
    else:
        from transformers import AutoModelForTokenClassification

        model = AutoModelForTokenClassification.from_pretrained(path)

    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception:  # noqa: BLE001
        tok = AutoTokenizer.from_pretrained(backbone)

    return Bundle(
        model=model.eval().to(device),
        tokenizer=tok,
        name=run_name,
        backbone=backbone,
        max_length=max_length,
        is_lora=is_lora,
        device=device,
        metrics=metrics,
    )


def analyze(
    bundle: Bundle,
    question: str,
    context: str,
    answer: str,
    threshold: float = cfg.SPAN_THRESHOLD,
    with_baseline: bool = True,
) -> dict:
    """Run the model on one triple and, optionally, the lexical baseline too.

    Showing the baseline beside the model in the demo is the same discipline
    the results tables use: it makes visible how much of the verdict needed a
    model at all.
    """
    t0 = time.perf_counter()
    out = predict_one(
        bundle.model,
        bundle.tokenizer,
        question=question,
        context=context,
        answer=answer,
        device=bundle.device,
        max_length=bundle.max_length,
        threshold=threshold,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "run": bundle.name,
        "backbone": bundle.backbone,
        "is_lora": bundle.is_lora,
        "untrained": bundle.name == "UNTRAINED",
        "hallucinated": out["hallucinated"],
        "score": out["score"],
        "words": out["words"],
        "word_probs": out["word_probs"],
        "word_labels": out["word_labels"],
        "spans": [{"start": s, "end": e, "text": t} for s, e, t in out["spans"]],
        "latency_ms": latency_ms,
        "threshold": threshold,
    }

    if with_baseline:
        bl = LexicalOverlapBaseline(mode="morph")
        preds = bl.predict_words(context, out["words"])
        result["baseline"] = {
            "name": bl.name,
            "hallucinated": bool(any(preds)),
            "score": bl.score(preds),
            "word_labels": preds,
            "spans": [t for _, _, t in ev.extract_spans(out["words"], preds)],
        }

    return result
