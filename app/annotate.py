"""Annotation UI for the human-verified gold test set.

    streamlit run app/annotate.py

Each teammate picks their name, reads context, question and answer, clicks the
answer words that are not supported by the context, and gives a verdict. The
generator's label is never shown: annotation is blind. Work saves to
data/gold/annotations/<name>.jsonl, one file per person, so everyone can commit
their own file without conflicts.

Needs only pandas and Streamlit -- no torch.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import gold_io as gio  # noqa: E402

st.set_page_config(page_title="Gold set annotation", page_icon="🏷️", layout="wide")

GUIDELINES = """
**Your job.** Decide whether the answer says anything the context does not support,
and if so, mark exactly which words.

1. **The context is the only truth.** Ignore what you know about the world. A statement
   that is true in real life but not stated in the context is still *unsupported*.
2. **Mark the smallest set of words that is wrong**, usually 1–4 words: the wrong name,
   number, date, or the added claim. Don't mark the correct words around it.
3. **Contradiction:** the answer says the opposite of the context, or states it wrongly.
   Mark the words that make it wrong.
4. **Added information:** the answer adds a fact the context doesn't contain. Mark the
   added part.
5. **Omission:** the answer presents an incomplete list or claim as if complete, e.g.
   naming two items when the context gives three. Mark the incomplete part.
6. **Paraphrase is fine.** Different wording, synonyms, `ও` vs `এবং`, or small grammar
   changes are *supported* if the meaning matches the context.
7. **Punctuation** (`।`, `,`) is never marked on its own.
8. Choose **Cannot decide** only if the context truly doesn't let a careful reader decide.
   Use the note field to say why.

Work alone. Don't look at the dataset CSV or discuss items until everyone has finished,
because agreement between annotators is one of the things being measured.
"""


def render_words(words: list[str], flagged: set[int], colour: str = "#f8b4b4") -> str:
    parts = []
    for i, w in enumerate(words):
        esc = html.escape(w)
        if i in flagged:
            parts.append(f"<span style='background:{colour};border-radius:4px;padding:1px 3px'>{esc}</span>")
        else:
            parts.append(esc)
    return " ".join(parts)


def item_view(item) -> None:
    st.markdown(f"**Question**  \n{item['question']}")
    with st.container(border=True):
        st.markdown("**Context** (the only source of truth)")
        st.write(item["context"])
    st.markdown(f"**Answer**  \n{item['answer']}")


def annotate(me: str) -> None:
    items = gio.load_items()
    mine = items[items["annotators"].map(lambda x: me in x)].sort_values("position").reset_index(drop=True)
    done = gio.load_annotations(me)
    n_done = sum(1 for i in mine["id"] if i in done)
    st.progress(n_done / max(len(mine), 1), text=f"{n_done} / {len(mine)} done")

    key = f"idx-{me}"
    if key not in st.session_state:
        pending = [k for k, i in enumerate(mine["id"]) if i not in done]
        st.session_state[key] = pending[0] if pending else 0
    idx = st.session_state[key]

    nav = st.columns([1, 1, 1, 3])
    if nav[0].button("◀ Previous", disabled=idx == 0, use_container_width=True):
        st.session_state[key] = idx - 1
        st.rerun()
    if nav[1].button("Next ▶", disabled=idx >= len(mine) - 1, use_container_width=True):
        st.session_state[key] = idx + 1
        st.rerun()
    if nav[2].button("Next unfinished", use_container_width=True):
        pending = [k for k, i in enumerate(mine["id"]) if i not in done]
        if pending:
            st.session_state[key] = next((k for k in pending if k > idx), pending[0])
            st.rerun()
    nav[3].caption(f"Item {idx + 1} of {len(mine)} · id `{mine.loc[idx, 'id']}`")

    item = mine.loc[idx]
    prev = done.get(item["id"])
    if prev:
        st.success(f"Already saved ({gio.VERDICTS[prev['verdict']]}). You can change it and save again.")
    item_view(item)

    words = item["words"]
    flagged = st.pills(
        "Click the answer words that are NOT supported by the context",
        options=list(range(len(words))),
        format_func=lambda i: words[i],
        selection_mode="multi",
        default=prev["flagged"] if prev else [],
        key=f"pills-{me}-{item['id']}",
    )
    verdict_keys = list(gio.VERDICTS)
    verdict = st.radio(
        "Verdict",
        verdict_keys,
        format_func=lambda v: gio.VERDICTS[v],
        index=verdict_keys.index(prev["verdict"]) if prev else None,
        horizontal=True,
        key=f"verdict-{me}-{item['id']}",
    )
    note = st.text_input("Note (optional)", value=prev.get("note", "") if prev else "", key=f"note-{me}-{item['id']}")

    if flagged:
        st.markdown("Marked: " + render_words(words, set(flagged)), unsafe_allow_html=True)

    if st.button("💾 Save and go to next", type="primary", use_container_width=True):
        if verdict is None:
            st.error("Choose a verdict.")
        elif verdict == "hallucinated" and not flagged:
            st.error("You chose *Contains a hallucination*. Mark at least one word.")
        elif verdict == "supported" and flagged:
            st.error("You marked words but chose *Fully supported*. Clear the words or change the verdict.")
        else:
            gio.save_annotation(me, item["id"], verdict, list(flagged) if verdict == "hallucinated" else [], note)
            if idx < len(mine) - 1:
                st.session_state[key] = idx + 1
            st.rerun()


def adjudicate(me: str) -> None:
    items = gio.load_items().set_index("id", drop=False)
    ann = gio.load_all_annotations()
    queue = gio.needs_adjudication(items.reset_index(drop=True), ann)
    settled = gio.load_adjudicated()
    open_items = [i for i in queue if i not in settled]
    st.info(
        f"{len(queue)} disagreements, {len(queue) - len(open_items)} settled. Settle them together, "
        "after both annotators have finished, by agreeing on the answer the guidelines support."
    )
    if not queue:
        return
    pick = st.selectbox(
        "Disagreement",
        queue,
        format_func=lambda i: f"{i}  {'✅ settled' if i in settled else '⏳ open'}",
    )
    item = items.loc[pick]
    item_view(item)
    words = item["words"]
    cols = st.columns(2)
    for col, a in zip(cols, item["annotators"]):
        rec = ann[a][pick]
        col.markdown(f"**{a}**: {gio.VERDICTS[rec['verdict']]}")
        col.markdown(render_words(words, set(rec["flagged"])), unsafe_allow_html=True)
        if rec.get("note"):
            col.caption(f"note: {rec['note']}")

    prev = settled.get(pick)
    flagged = st.pills(
        "Final: words not supported by the context",
        options=list(range(len(words))),
        format_func=lambda i: words[i],
        selection_mode="multi",
        default=prev["flagged"] if prev else [],
        key=f"adj-pills-{pick}",
    )
    verdict_keys = list(gio.VERDICTS)
    verdict = st.radio(
        "Final verdict",
        verdict_keys,
        format_func=lambda v: gio.VERDICTS[v],
        index=verdict_keys.index(prev["verdict"]) if prev else None,
        horizontal=True,
        key=f"adj-verdict-{pick}",
    )
    note = st.text_input("Reason", value=prev.get("note", "") if prev else "", key=f"adj-note-{pick}")
    if st.button("💾 Save decision", type="primary"):
        if verdict is None:
            st.error("Choose a verdict.")
        elif verdict == "hallucinated" and not flagged:
            st.error("Mark at least one word.")
        else:
            gio.save_adjudication(pick, verdict, list(flagged) if verdict == "hallucinated" else [], me, note)
            st.rerun()


st.title("🏷️ Gold set annotation")
with st.sidebar:
    me = st.selectbox("Who are you?", gio.ANNOTATORS, index=None, placeholder="Choose your name")
    mode = st.radio("Mode", ["Annotate", "Adjudicate"], help="Adjudicate only after everyone has finished.")
    with st.expander("Guidelines", expanded=me is None):
        st.markdown(GUIDELINES)

if me is None:
    st.info("Choose your name in the sidebar to start. Read the guidelines first.")
elif mode == "Annotate":
    annotate(me)
else:
    adjudicate(me)
