"""Single place to configure a run.

CLAUDE.md asks for one training script driven by a config, so the backbone is a
string here and every candidate can be run without editing code elsewhere.

Note on the dataset: the CSV currently in data/ is a SAMPLE, not the final
dataset -- the real one is being built separately. Nothing here hardcodes that
sample's dimensions. In particular MAX_LENGTH defaults to "auto", which measures
the loaded data rather than assuming the sample's very short answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CSV_PATH = DATA_DIR / "bangla_rag_halu_4000.csv"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"


# --------------------------------------------------------------------------
# Dataset schema
#
# Kept as a dataclass rather than inline string literals so that a rename in the
# real dataset is a one-line change here instead of a hunt through data.py.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSchema:
    id: str = "id"
    domain: str = "domain"
    context: str = "context"
    question: str = "question"
    answer: str = "answer"
    label: str = "label"
    hallucination_type: str = "hallucination_type"
    hallucinated_span: str = "hallucinated_span"
    token_labels: str = "token_labels"
    #: Rows sharing a value here share a context and must never be split apart.
    #: When the column is absent (the old sample, the RAGTruth adapter) the
    #: group key is derived from the id via ID_SUFFIX_PATTERN instead.
    group: str = "context_id"

    @property
    def required(self) -> list[str]:
        """Columns data.py refuses to proceed without."""
        return [
            self.id,
            self.context,
            self.question,
            self.answer,
            self.label,
            self.token_labels,
        ]


COLUMNS = ColumnSchema()

#: Fallback group key for datasets without a ``group`` column: strips the
#: faithful/hallucinated twin marker off an id. ``BD_HIS_01519_F`` ->
#: ``BD_HIS_01519``. The anti-leakage guarantee rests on the group key, so
#: data.py verifies it actually groups something.
ID_SUFFIX_PATTERN = r"_(F|H)$"

#: How an answer string is cut into the words that carry labels. Inference must
#: cut raw answers exactly as the dataset generator did, or predicted words and
#: gold words stop lining up.
#:
#: The 4,000-row dataset splits punctuation (the danda, commas, brackets) into
#: its own tokens but keeps numbers like ``১০,০০০`` and ``২.৫`` whole, and treats
#: only Bengali and ASCII characters as word characters. This pattern
#: reproduces its ``token_labels`` for every row. data.py detects which scheme a
#: dataset uses ("regex" or "whitespace") and fails if neither fits.
ANSWER_TOKEN_PATTERN = (
    r"[\u09E6-\u09EF0-9]+(?:[.,/:][\u09E6-\u09EF0-9]+)+"
    r"|[\u0980-\u09FFA-Za-z0-9\u200c\u200d]+"
    r"|\S"
)

#: Rows whose context/answer contain no Bengali script. CLAUDE.md is explicit
#: that these must not be dropped silently: either keep and note, or drop and
#: note. Default keeps them; data.py reports the count either way.
DROP_NON_BANGLA = False


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------

#: The only line most runs need to change.
BACKBONE = "csebuetnlp/banglabert"


@dataclass(frozen=True)
class BackboneSpec:
    hf_id: str
    note: str
    #: LoRA injection points differ by architecture; None lets peft guess.
    lora_targets: tuple[str, ...] | None = None
    #: Hard ceiling from the model's position embeddings.
    model_max_length: int = 512


BACKBONES: dict[str, BackboneSpec] = {
    "csebuetnlp/banglabert": BackboneSpec(
        hf_id="csebuetnlp/banglabert",
        note="ELECTRA-base ~110M. Primary. Best Bangla fertility measured (1.40 "
        "subwords/word) but only 512 positions -- watch for truncation once the "
        "real dataset's contexts get longer.",
        lora_targets=("query", "key", "value"),
        model_max_length=512,
    ),
    "jhu-clsp/mmBERT-base": BackboneSpec(
        hf_id="jhu-clsp/mmBERT-base",
        note="Multilingual ModernBERT, 307M total / 110M non-embedding, 8192 "
        "positions. NOT the English ModernBERT that CLAUDE.md rules out. Bangla "
        "fertility measured at 3.99 subwords/word -- ~2.9x BanglaBERT, so it "
        "costs sequence length and latency, but it cannot run out of context.",
        lora_targets=("Wqkv",),
        model_max_length=8192,
    ),
    "microsoft/mdeberta-v3-base": BackboneSpec(
        hf_id="microsoft/mdeberta-v3-base",
        note="~278M. Keeps the DeBERTa thread from the FYDP-I paper. Fertility 2.82.",
        lora_targets=("query_proj", "key_proj", "value_proj"),
        model_max_length=512,
    ),
    "xlm-roberta-base": BackboneSpec(
        hf_id="xlm-roberta-base",
        note="278M. Comparison point. Fertility 2.20.",
        lora_targets=("query", "key", "value"),
        model_max_length=512,
    ),
    "google/muril-base-cased": BackboneSpec(
        hf_id="google/muril-base-cased",
        note="~238M. Indic-focused comparison point.",
        lora_targets=("query", "key", "value"),
        model_max_length=512,
    ),
}


def backbone_spec(hf_id: str | None = None) -> BackboneSpec:
    hf_id = hf_id or BACKBONE
    if hf_id not in BACKBONES:
        # Unknown backbones are allowed; peft infers targets and we assume 512.
        return BackboneSpec(hf_id=hf_id, note="not in registry", lora_targets=None)
    return BACKBONES[hf_id]


# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------

#: "auto" measures the loaded dataset and picks a length covering
#: AUTO_LENGTH_PERCENTILE of examples, rounded up to a multiple of 32 and capped
#: at AUTO_LENGTH_CAP. Set an int to pin it.
#:
#: CLAUDE.md pins 256 and says not to pad to 512 "just in case". That is correct
#: for the sample (p95 is 68 subwords under BanglaBERT) but will be wrong once
#: answers become 3-5 sentences, so the default measures instead of assuming.
MAX_LENGTH: int | str = "auto"
AUTO_LENGTH_PERCENTILE = 0.99
#: Only a ceiling for backbones with more positions; each backbone's own limit
#: (BanglaBERT 512) still applies. mmBERT needs ~864 to cover p99 of the
#: 4,000-row dataset -- at 512 it would truncate context on ~19% of rows.
AUTO_LENGTH_CAP = 1024
AUTO_LENGTH_MULTIPLE = 32

#: Label only the first subword of each answer word; the rest get -100.
#: Flipping this to True labels every subword and changes token-level metrics,
#: so it must be reported alongside results either way.
LABEL_ALL_SUBWORDS = False

#: Ignored by the loss. Questions, contexts and special tokens all get this.
IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

#: Grouped on the context so its faithful and hallucinated rows can never
#: land on opposite sides of a split.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLIT_SEED = 42

#: Stratify group assignment by domain so a small domain cannot vanish from a
#: split (Bangladesh Affairs had only 44 rows in the first sample).
STRATIFY_BY_DOMAIN = True


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    backbone: str = BACKBONE
    epochs: int = 4
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    #: LoRA usually wants a markedly higher LR than full fine-tuning.
    lora_learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = False  # CPU-only venv; enable on a CUDA box.

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    #: Positive-class weight in the token loss. Answers are mostly grounded even
    #: in hallucinated rows, so positives are the minority at token level.
    positive_class_weight: float = 1.0

    #: Optimizer steps every N batches, so the effective batch is
    #: train_batch_size * grad_accum_steps. Lets a long-sequence backbone
    #: (mmBERT at ~860 subwords) keep an effective batch of 16 in 16GB RAM.
    grad_accum_steps: int = 1
    #: Recompute activations in the backward pass: ~30% slower, far less memory.
    gradient_checkpointing: bool = False

    early_stopping_patience: int = 2
    metric_for_best_model: str = "token_f1"

    extra: dict = field(default_factory=dict)


TRAIN = TrainConfig()


# --------------------------------------------------------------------------
# Inference / evaluation
# --------------------------------------------------------------------------

#: Merge runs of consecutive tokens above this into highlighted spans.
SPAN_THRESHOLD = 0.5

#: Example-level score is the max over answer-token hallucination probabilities.
EXAMPLE_AGGREGATION = "max"

#: Latency must be measured locally, batch size 1, after warmup. Never on Colab.
LATENCY_WARMUP_RUNS = 20
LATENCY_MEASURED_RUNS = 100
LATENCY_BATCH_SIZE = 1
