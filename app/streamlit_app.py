"""Streamlit demo: paste a context, question and answer, see the hallucinated
spans highlighted.

    streamlit run app/streamlit_app.py

The lexical baseline's verdict is shown beside the model's on purpose. It is
the same honesty the results tables require -- if a five-line string match
flags the same words, the demo should show that rather than imply the model
did something only a model could do.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app._loader import Bundle, analyze, find_checkpoints, load_bundle  # noqa: E402

st.set_page_config(page_title="Bangla RAG Hallucination Detector", page_icon="🔍", layout="wide")

EXAMPLE = {
    "question": "বাংলাদেশের রাজধানী কোথায়?",
    "context": "বাংলাদেশের রাজধানী ঢাকা। এটি দেশের বৃহত্তম শহর এবং প্রধান বাণিজ্যিক কেন্দ্র।",
    "answer": "বাংলাদেশের রাজধানী চট্টগ্রাম।",
}


@st.cache_resource(show_spinner="Loading model…")
def _load(ckpt: str | None) -> Bundle:
    return load_bundle(ckpt)


def highlight(words: list[str], probs: list[float], labels: list[int]) -> str:
    parts = []
    for w, p, lab in zip(words, probs, labels):
        safe = (
            str(w).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        if lab:
            alpha = 0.25 + 0.55 * min(max(p, 0.0), 1.0)
            parts.append(
                f'<span style="background:rgba(220,38,38,{alpha:.2f});'
                f'border-bottom:2px solid #dc2626;padding:1px 3px;border-radius:3px" '
                f'title="p={p:.3f}">{safe}</span>'
            )
        else:
            parts.append(f'<span title="p={p:.3f}">{safe}</span>')
    return " ".join(parts)


st.title("🔍 Bangla RAG Hallucination Detector")
st.caption(
    "Token-level detection with a small encoder. Runs locally — no API calls, no cloud inference."
)

# ---- sidebar ---------------------------------------------------------------
ckpts = find_checkpoints()
with st.sidebar:
    st.header("Model")
    choice = None
    if ckpts:
        names = [p.name for p in ckpts]
        picked = st.selectbox("Checkpoint", names, index=0)
        choice = str(ckpts[names.index(picked)])
    else:
        st.warning("No checkpoint in outputs/checkpoints. Using the untrained backbone.")
    threshold = st.slider("Span threshold", 0.05, 0.95, 0.5, 0.05)
    show_baseline = st.checkbox("Compare with lexical baseline", value=True)

bundle = _load(choice)

with st.sidebar:
    st.divider()
    st.markdown(
        f"**Run:** `{bundle.name}`  \n"
        f"**Backbone:** `{bundle.backbone}`  \n"
        f"**Tuning:** {'LoRA' if bundle.is_lora else 'full'}  \n"
        f"**max_length:** {bundle.max_length}"
    )
    test = (bundle.metrics.get("test") or {}) if bundle.metrics else {}
    if test:
        w, s = test.get("word_level", {}), test.get("span_level", {})
        st.markdown(
            f"**Test word F1:** {w.get('f1', float('nan')):.3f}  \n"
            f"**Test span exact:** {s.get('exact', float('nan')):.3f}"
        )

if bundle.name == "UNTRAINED":
    st.error(
        "No trained checkpoint was found, so this is a randomly initialised "
        "classification head. **Predictions below are meaningless.** Train one with "
        "`python -m src.train --save-model`."
    )

# ---- input -----------------------------------------------------------------
left, right = st.columns(2)
with left:
    question = st.text_area("Question", EXAMPLE["question"], height=80)
    answer = st.text_area("LLM answer (checked for hallucination)", EXAMPLE["answer"], height=120)
with right:
    context = st.text_area("Retrieved context (the ground truth)", EXAMPLE["context"], height=224)

if st.button("Detect hallucinations", type="primary", width="stretch"):
    if not answer.strip():
        st.warning("Enter an answer to check.")
        st.stop()

    res = analyze(bundle, question, context, answer, threshold=threshold, with_baseline=show_baseline)

    verdict, score = res["hallucinated"], res["score"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Verdict", "Hallucinated" if verdict else "Grounded")
    c2.metric("Max token probability", f"{score:.3f}")
    c3.metric("Latency", f"{res['latency_ms']:.0f} ms")

    st.subheader("Model")
    st.markdown(
        f"<div style='font-size:1.15rem;line-height:2.1'>"
        f"{highlight(res['words'], res['word_probs'], res['word_labels'])}</div>",
        unsafe_allow_html=True,
    )
    if res["spans"]:
        st.caption("Flagged spans: " + " · ".join(f"“{s['text']}”" for s in res["spans"]))
    else:
        st.caption("No span exceeded the threshold.")

    if show_baseline and "baseline" in res:
        b = res["baseline"]
        st.subheader(f"Lexical baseline ({b['name']})")
        st.markdown(
            f"<div style='font-size:1.05rem;line-height:2;opacity:.85'>"
            f"{highlight(res['words'], [float(x) for x in b['word_labels']], b['word_labels'])}</div>",
            unsafe_allow_html=True,
        )
        agree = b["hallucinated"] == verdict
        st.caption(
            ("Baseline agrees with the model at example level."
             if agree else
             "Baseline **disagrees** with the model at example level.")
            + f" Baseline flagged {sum(b['word_labels'])} of {len(res['words'])} words."
        )

    with st.expander("Per-word probabilities"):
        st.dataframe(
            {
                "word": res["words"],
                "p(hallucinated)": [round(p, 4) for p in res["word_probs"]],
                "flagged": res["word_labels"],
            },
            width="stretch",
        )
