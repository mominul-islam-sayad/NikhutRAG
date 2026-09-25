# FYDP-II — Bangla RAG Hallucination Detection

Context file for Claude Code. Read this before writing any code.

## What this project is

**"An Efficient and Lightweight Hallucination Detection Framework for RAG Using SLMs"**
United International University, CSE, Group B2.
Team: Mehrin Sultana, Mominul Islam, Nazmul Alam, Md. Mainul Hossain Fahim Chowdhury.
Supervisor: Khushnur Binte Jahangir. Co-supervisor: Nahid Hossain.

FYDP-I delivered the paper, poster and design. FYDP-II is the implementation.

**The method:** encode `context + question + answer` jointly in an encoder-only small
language model, run **token-level binary classification over the answer tokens only**
(0 = grounded in context, 1 = hallucinated), then aggregate consecutive positive tokens
into highlighted spans shown to the user. This is deliberately *not* LLM-as-a-judge —
the whole contribution is that it's small, local, fast and free.

## The paper says RAGTruth, not Bangla — resolved 2026-09-18

`docs/Fydp1_final_paper.pdf` commits to the **English RAGTruth corpus** and does not
contain the words Bangla, Bengali, Bangladesh, Indic or multilingual anywhere in its 26
pages. RAGTruth is named 14+ times, and task allocation marks "Dataset Acquisition
(RAGTruth)" and "Preprocessing RAGTruth Data" as 100% complete.

**Decision: Bangla is primary, plus a small English RAGTruth arm.**

Reasons, in the order they matter:

- The paper names **Luna as "the main baseline"** and **LettuceDetect (F1 79.22% on
  RAGTruth)** as the system to beat. Neither exists in Bangla. Without a RAGTruth arm,
  every comparison point in the literature review becomes incomparable.
- The paper's own **gap analysis row 4** already states: *"Absence of bilingual illusion
  identification abilities. The majority of current datasets and detectors are
  exclusively in English."* Bangla delivers that stated gap. **Update that row to cite
  Bangla rather than Chinese** — it converts the pivot from a deviation into the
  contribution.
- The paper text must be revised to say all of this. It currently does not.

## Hard constraints (commitments from FYDP-I, not preferences)

- Model **under 500M parameters** — BanglaBERT measured at 110,028,290. Satisfied.
- Inference **under 200ms** per example. The paper's wording is stricter: "within
  100-200 milliseconds for each inference" (p.15). Measured 34.8ms median on CPU.
- **LoRA / PEFT** fine-tuning must be used and measured
- Runs **locally** on consumer hardware — no API calls, no cloud inference
- Token-level output, not just a sequence-level true/false
- **Streamlit UI** demo, **and a FastAPI + Uvicorn backend** — the paper's software
  stack (p.17) commits to both. Streamlit alone does not satisfy the paper.

## Headline metric: word-level and span-level F1 — decided 2026-09-18

The paper sets F1 > 75%. **That target is 19 points below a five-line string
comparison** (see below) and must be raised, and the unit must change.

- **Primary metrics: word-level F1 and span-level exact match.** These are what the UI
  highlights and what the paper's "Span Aggregation" requirement is about.
- **Example-level F1 is reported only with the lexical baseline beside it.** Alone it is
  misleading: on a 1-epoch smoke test the model beat the fair baseline by just +0.036 at
  example level but +0.241 at word level and +0.236 at span level.
- Set the new target against word/span F1 once a tuned run exists. Do not carry >75%
  forward as an example-level claim.

## The dataset

`data/bangla_rag_halu_4000.csv` (committed 2026-09-25) replaced the first 2,652-row
sample `bangla_rag_hallucination_8k.csv`. It is still being built outside this repo and
may change again — do not hardcode any of it. `src/data.py` validates and measures
instead, and prints all of the checks below on every load.

| Property | Value (4,000-row dataset) |
|---|---|
| Rows | 4,000 (2,000 faithful, 2,000 hallucinated — exactly 50/50) |
| Contexts (`context_id`) | 943, with **2, 4 or 6 rows each** (230 / 369 / 344 contexts) |
| Structure | each context+question has one faithful and one hallucinated answer; every context has both labels |
| IDs | `BDA_0001`, `HIS_0050`, … — **no `_F`/`_H` suffix any more** |
| `token_labels` format | `[{"token": w, "label": l}, …]`, punctuation split off as its own token |
| Answer length | 15 words median, p95 27, max 51; **96% are a single sentence** |
| Context length | 71 words median, max 219 |
| Rows lacking Bengali script | 0 |
| Rows with foreign-script letters | 23 — 3 are real corruption inside an answer word (`কবিতայ`, `আওत`, `सार्वजनिक`), 20 are in contexts |
| Human verified | 0% |
| Source | Bangla Wikipedia 3,658, DGHS 162, National Portal 126, Ministry of Education 48, BANBEIS 6 |

Domains: 500 each of Bangladesh Affairs, History, Healthcare, Agriculture, Government
Services, Education, Universities, Science & Technology.

Hallucination types: `fabricated_info` 334, `entity_replacement` 334, `omission` 333,
`contradiction` 333, `number_error` 333, `date_error` 333, `none` 2000. Note `omission`
is new and `fabricated_fact` was renamed `fabricated_info`.

`config.DROP_NON_BANGLA` still exists but currently affects no rows.

## Three things that must not be gotten wrong

### 1. Split leakage

Each context appears 2, 4 or 6 times, under different questions and with faithful and
hallucinated answers. A random row-level split puts the same context on both sides and
the reported F1 becomes meaningless.

**Always group-split on `context_id`** (`ColumnSchema.group`). Grouping on the ID alone
would now be **wrong**: IDs are unique per row, so the old `_F`/`_H` suffix rule would
leak. The loader falls back to the suffix rule only when the column is missing (old
sample, RAGTruth adapter), and it refuses to load if one context text appears under two
`context_id`s or one `context_id` holds two texts.

Implemented in `src/data.py:group_split`, which groups *and* stratifies by domain in one
pass and asserts the splits are disjoint. On the 4,000-row dataset: train 2,824 rows /
659 contexts, val 588 / 142, test 588 / 142, all eight domains in every split,
positive rate 0.500 in each.

### 2. The lexical baseline — always in the table

Flagging any answer word absent from the context — no model, no training. On the first
sample it scored example F1 0.939 (`morph`), above the paper's >75% target. On the
**4,000-row dataset it collapses**, because in 24% of hallucinated rows every flagged
word also appears verbatim somewhere in the context:

| mode | split | ex_P | ex_R | ex_F1 | ex_Acc | AUROC | word_F1 | span_F1 | span_exact | span_partial |
|---|---|---|---|---|---|---|---|---|---|---|
| `exact` | test (588) | 0.523 | 0.952 | 0.676 | 0.543 | 0.641 | 0.333 | 0.072 | 0.207 | 0.820 |
| `morph` | test (588) | 0.549 | 0.915 | **0.686** | 0.582 | 0.659 | **0.370** | 0.089 | 0.211 | 0.731 |
| `morph` | all (4,000) | 0.553 | 0.907 | 0.687 | 0.586 | 0.674 | 0.396 | 0.094 | 0.216 | 0.744 |

For comparison, the first sample (morph, all rows): ex_F1 0.939, word_F1 0.581.

`exact` is **handicapped by Bangla morphology** — `গ্রিসে` does not string-match `গ্রিস`.
`morph` absorbs case suffixes and is **the honest bar**. Report `morph`.

Both live in `src/baselines.py` and **must appear in every results table**;
`src/train.py` prints them automatically beside the model row.

Per type (morph, test): `omission` is nearly invisible to it (ex recall 0.587, word F1
0.113), as is `entity_replacement` at word level (0.198). `fabricated_info` is its easiest
case (word F1 0.756).

### 3. Words at inference must be cut exactly as the labels were

The labelled words are the tokens in `token_labels`, **not** `answer.split()`. The
4,000-row dataset splits punctuation (`।`, `,`, `(`, `-`, `%`) into its own tokens but
keeps numbers whole (`১০,০০০`, `২.৫`, `1.4.1.2`), and treats only Bengali and ASCII as
word characters. `config.ANSWER_TOKEN_PATTERN` reproduces its tokens for **all 4,000
rows**. `src/data.py:tokenize_answer` applies it.

The loader detects which scheme a dataset uses (`regex` or `whitespace`), refuses to load
if neither fits every row, and `src/train.py` saves it as `word_scheme` in the metrics
JSON. `app/_loader.py` reads it back, so the UI and API cut unseen answers the same way
the checkpoint was trained. Runs with no `word_scheme` are treated as `whitespace`.

**The first sample's generator bug is fixed in this dataset.** The sample had 66
hallucinated rows (5.0%) with no flagged token and 316 (23.8%) whose flagged words
disagreed with `hallucinated_span` — exact whitespace matching broke on inflection and
the attached danda. The 4,000-row dataset has **0** of either: flagged tokens match
`hallucinated_span` on every row once whitespace is ignored. `src/data.py` still reports
both counts on every load, in case a regeneration brings the bug back.

## Planned dataset regeneration

Status against the 4,000-row dataset:

1. Multi-sentence answers (3–5 sentences) where only **one clause** is corrupted —
   **not yet**: 96% of answers are still one sentence (15 words median).
2. Corruptions that **reuse vocabulary present elsewhere in the context**, so lexical
   overlap cannot detect them — **partly**: in 24% of hallucinated rows every flagged
   word appears verbatim in the context, and the baseline fell from 0.939 to 0.686 ex_F1.
3. A **human-verified gold test set of ~200 examples**, annotated by the team — **not yet**.
4. A second test set at a **realistic class ratio**. Note: RAGTruth is **not** 15–20% —
   measured, **43.1%** of its 17,790 responses carry at least one hallucination span.
   Pick the target ratio from the deployment story being argued, not from RAGTruth, and
   report it alongside the balanced set — **not yet**; the dataset is again exactly 50/50.
5. Fix the span-matching bug — **done** (0 unlearnable rows, 0 span mismatches).
6. Clean the 3 answers with foreign-script letters inside Bangla words (reported on load).

Sequence length already grew: p99 is 288 subwords (was 86), max 365, so `max_length`
auto-selects 288. That still fits BanglaBERT's 512 positions, but 3–5 sentence answers
could push it past; the loader reports truncated contexts when it happens.

## Model choice

**English ModernBERT is not for Bangla** — its tokenizer fragments Bangla badly. Use it
**only** for the English RAGTruth arm.

Measured Bangla tokenizer fertility (subwords per word, lower is better), on 300 rows of
the actual dataset:

| backbone | params | vocab | fertility | positions |
|---|---|---|---|---|
| `csebuetnlp/banglabert` | 110M | 32k | **1.40** | 512 |
| `xlm-roberta-base` | 278M | 250k | 2.20 | 512 |
| `microsoft/mdeberta-v3-base` | 278M | 250k | 2.82 | 512 |
| `jhu-clsp/mmBERT-base` | 307M | 256k | 3.99 | **8192** |

1. `csebuetnlp/banglabert` — **primary.** Best fertility by ~2.9×. Its 512-position
   ceiling is the risk once contexts lengthen.
2. `jhu-clsp/mmBERT-base` — **main comparison.** This is the *multilingual* ModernBERT,
   not the English one, so it keeps the paper's ModernBERT thread intact honestly.
   Higher fertility costs sequence length and latency, but it cannot run out of context.
   Latency headroom exists: BanglaBERT measured 34.8ms against a 200ms budget.
   `AUTO_LENGTH_CAP` currently caps it at 512 — raise it to use its long context.
3. `microsoft/mdeberta-v3-base`, `xlm-roberta-base`, `google/muril-base-cased` —
   additional comparison points.
4. `answerdotai/ModernBERT-base` — English RAGTruth arm only, never for Bangla.

One training script driven by config: the backbone is a string in `src/config.py`.

## Input encoding and labels

- Sequence: `[CLS] question [SEP] context [SEP] answer [SEP]`
- **`max_length` is measured, not pinned.** `config.MAX_LENGTH = "auto"` covers the 99th
  percentile, rounded to a multiple of 32, capped by `AUTO_LENGTH_CAP` and the backbone's
  position limit. On the 4,000-row dataset this selects **288** (p99 = 288, median 135);
  on the first sample it selected 96.
- Tokenize the **answer with `is_split_into_words=True`** on the `token_labels` words
  (see "Words at inference" above), so `word_ids()` maps subwords back to the labels.
  Never reconstruct from char offsets.
- Label **only answer subword tokens**; question, context and specials get `-100`.
- Label the **first subword** of each word, rest `-100` (`LABEL_ALL_SUBWORDS=False`).
- **Truncation is `only_first`** — sacrifice context before the answer. Truncating the
  answer silently discards labels, which is unrecoverable.
- Example-level score = **max** over answer token hallucination probabilities.
- Span extraction = merge runs of consecutive tokens above threshold 0.5.

## Evaluation protocol

All implemented in `src/evaluate.py`, which scores every system through one code path.

- **Word-level** P/R/F1 on the hallucinated class — *headline*
- **Span-level** — *headline*. Gold spans are runs of flagged words in the gold token
  labels; spans are compared by word position, not text. Reported as span F1 (exact
  boundaries, pooled over all rows so spans on faithful answers count as false
  positives), span exact (share of hallucinated rows whose every gold span is predicted
  exactly) and span partial (any overlap). Text comparison against `hallucinated_span`
  was dropped once punctuation became its own word; the two agree on every row anyway.
- **Example-level** P/R/F1, accuracy, AUROC — only with the baseline beside it
- **Per hallucination type** — types are now balanced (~333 each, 41–59 per type in
  test). `omission` is the new one and the baseline's weakest; say out loud whichever
  type the model is weakest on.
- **Latency**: ms per example at batch size 1, CPU and GPU separately, after ~20 warmups,
  on the actual local machine. Never on Colab — the claim is about consumer hardware.
- **Efficiency**: parameter count, trainable count under LoRA, peak memory, wall-clock

Compare against: the lexical baseline (both modes), full fine-tuning vs LoRA, and at
least one LLM-as-a-judge reference point for the cost argument. For the RAGTruth arm,
compare to Luna and LettuceDetect's published numbers.

### LoRA vs full fine-tuning — first sample, measured 2026-09-18

**To be superseded by the 4,000-row rerun (in progress); kept for the record.** Measured on the first
2,652-row sample, whose baseline was far stronger, so none of these numbers transfer.
Both run, 4 epochs, BanglaBERT, CPU, identical splits and seed:

| | full | LoRA | delta |
|---|---|---|---|
| **word F1** | **0.857** | 0.831 | −0.026 |
| **span exact** | **0.667** | 0.661 | −0.006 |
| example F1 | 0.967 | **0.972** | +0.005 |
| AUROC | **0.997** | 0.992 | −0.005 |
| trainable params | 110,028,290 | **886,274** | 124× fewer |
| peak memory | 2,749 MB | **2,025 MB** | −26% |
| wall-clock | 1,123.9 s | **1,084.3 s** | −3.5% |
| latency (median) | **33.8 ms** | 37.2 ms | +3.4 ms |

**This is the legitimate finding the paper's LoRA claim needs.** At 110M, LoRA buys a
124× reduction in trainable parameters and 26% less memory, but **almost no training
time** — on CPU the backward pass still traverses the frozen backbone, so the saving
LoRA is famous for does not materialise at this scale. It costs 0.026 word F1, which is
the headline metric, while nudging example F1 up by 0.005 — a reminder that example-level
numbers are too coarse to rank these two.

Report this honestly rather than implying LoRA was necessary. It becomes genuinely
worthwhile on the larger backbones (mmBERT 307M, mDeBERTa 278M) and on GPU; re-measure
there before generalising from this row.

## Repo layout

```
.
├── CLAUDE.md
├── requirements.txt          # torch installed separately, CPU-only
├── .gitattributes
├── data/
├── docs/Fydp1_final_paper.pdf
├── src/
│   ├── config.py        # backbone, schema, hyperparams — the only file to edit per run
│   ├── data.py          # load, validate, group split, tokenize + label alignment
│   ├── model.py         # backbone + token classification head, LoRA wiring, inference
│   ├── train.py         # full vs LoRA; always prints the baseline beside the model
│   ├── evaluate.py      # word/example/span metrics, AUROC, latency, efficiency
│   └── baselines.py     # lexical string-match baseline, exact + morph
├── app/
│   ├── streamlit_app.py # not yet written
│   └── api.py           # FastAPI + Uvicorn, promised by the paper; not yet written
└── outputs/             # checkpoints, metrics JSON, plots (the paper's "Data Store D1")
```

## Environment

Local, VS Code, CPU. `.venv` holds CPU-only torch 2.14. 2,824 training rows, 177 steps
per epoch at `max_length` 288. No cloud compute needed for the primary
model. Colab is an escape hatch for mDeBERTa/mmBERT sweeps only — never for latency.

## Don't

- Don't random-split the rows. Group on base ID.
- Don't report example-level F1 as the headline, or without the baseline beside it.
- Don't omit the lexical baseline from results tables, and report the `morph` mode.
- Don't use English ModernBERT for Bangla. mmBERT-base is a different model.
- Don't benchmark latency on cloud hardware, or while another job is using the CPU.
- Don't silently drop the 10 non-Bangla rows — keep and note, or drop and note.
- Don't hardcode the sample's dimensions; the real dataset is being built elsewhere.
- Don't quote smoke-test numbers as results.
