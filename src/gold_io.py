"""Files and annotation records for the human-verified gold test set.

Kept free of torch and transformers so the annotation UI runs on any teammate's
machine with just pandas and Streamlit.

Layout under data/gold/:

    gold_items.csv                 the sample: id, context, question, answer,
                                   words (JSON), annotators (JSON), position
    annotations/<annotator>.jsonl  one file per person, append-only, so four
                                   people can commit without merge conflicts
    adjudicated.jsonl              final decisions on items the pair disagreed on

Every save appends a line; the latest line per item id wins, so an annotator
can revisit and change an answer without anything being overwritten.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "gold"
ITEMS_CSV = GOLD_DIR / "gold_items.csv"
ANNOTATION_DIR = GOLD_DIR / "annotations"
ADJUDICATED = GOLD_DIR / "adjudicated.jsonl"

#: The team (CLAUDE.md). Each gold item is annotated by two of them.
ANNOTATORS = ["Mehrin", "Mominul", "Nazmul", "Fahim"]

#: Verdicts an annotator can give. "unsure" means the context does not let a
#: careful reader decide; such items are excluded from scoring and counted.
VERDICTS = {
    "supported": "Fully supported by the context",
    "hallucinated": "Contains a hallucination",
    "unsure": "Cannot decide from the context",
}


def load_items() -> pd.DataFrame:
    if not ITEMS_CSV.exists():
        raise FileNotFoundError(f"{ITEMS_CSV} not found; run `python -m src.gold sample` first")
    df = pd.read_csv(ITEMS_CSV)
    df["words"] = df["words"].map(json.loads)
    df["annotators"] = df["annotators"].map(json.loads)
    return df


def _read_jsonl(path: Path) -> dict[str, dict]:
    """Latest record per item id."""
    latest: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                latest[rec["id"]] = rec
    return latest


def load_annotations(annotator: str) -> dict[str, dict]:
    return _read_jsonl(ANNOTATION_DIR / f"{annotator}.jsonl")


def load_all_annotations() -> dict[str, dict[str, dict]]:
    """annotator -> item id -> latest record."""
    return {a: load_annotations(a) for a in ANNOTATORS}


def load_adjudicated() -> dict[str, dict]:
    return _read_jsonl(ADJUDICATED)


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_annotation(annotator: str, item_id: str, verdict: str, flagged: list[int], note: str = "") -> None:
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    _append(
        ANNOTATION_DIR / f"{annotator}.jsonl",
        {"id": item_id, "annotator": annotator, "verdict": verdict, "flagged": sorted(flagged), "note": note},
    )


def save_adjudication(item_id: str, verdict: str, flagged: list[int], decided_by: str, note: str = "") -> None:
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    _append(
        ADJUDICATED,
        {"id": item_id, "verdict": verdict, "flagged": sorted(flagged), "decided_by": decided_by, "note": note},
    )


def pair_agrees(a: dict, b: dict) -> bool:
    """Two annotations agree when the verdict and the flagged words both match."""
    return a["verdict"] == b["verdict"] and (a["verdict"] != "hallucinated" or a["flagged"] == b["flagged"])


def needs_adjudication(items: pd.DataFrame, annotations: dict[str, dict[str, dict]]) -> list[str]:
    """Ids where both assigned annotators are done and they disagree."""
    out = []
    for _, it in items.iterrows():
        recs = [annotations[a].get(it["id"]) for a in it["annotators"]]
        if all(recs) and not pair_agrees(recs[0], recs[1]):
            out.append(it["id"])
    return out
