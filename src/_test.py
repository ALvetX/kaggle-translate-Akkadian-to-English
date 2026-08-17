#!/usr/bin/env python3
"""
Deep Past Challenge — Akkadian-to-English Translation
Ensemble MBR inference with corrected postprocessing (FIXED version)

Corrections applied based on host v3 update + community discussion:
  1. 5/12 shekel → ⅓ shekel 15 grains (was ⅔)
  2. Commodity regex respects word boundaries (no import-tax corruption)
  3. Parentheses preserved in translations (appear in test set)
  4. Curly quotes → straight quotes (not deleted)
  5. ḫ/Ḫ → h/H in postprocessing (not in test set)
  6. Safer slash-alternative regex (protects fractions)
  7. Stray mark removal (.., xx, <<>>, <>)
"""

# ======================================================================
# Better Candidate Diversity and Stronger Consensus Selection
# ======================================================================

# ======================================================================
# Notebook structure
# ======================================================================

# ======================================================================
# Corrections applied (based on host recommendations + community discussion)
# ======================================================================
import os, gc, re, json, math, random, logging, warnings
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import sacrebleu

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


# 1. Mixed precision helpers
#
# I use BF16 autocast whenever available.
# For ByT5 inference this is generally safer than FP16 while still reducing memory pressure.

def _cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False


def _fp16_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.get_device_properties(device).major >= 6
    except Exception:
        return False


def _amp_ctx(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if _cuda_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if _fp16_supported(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# 2. Configuration
#
# This configuration keeps the public notebook transparent:
#
# - two different ByT5 checkpoints,
# - standard beam search,
# - optional diverse beam search,
# - multi-temperature sampling,
# - weighted MBR reranking.

@dataclass
class EnsembleMBRConfig:
    test_data_path: str = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    output_dir: str = "/kaggle/working/"
    model_a_path: str = "/kaggle/input/mymodels"
    model_c_path: str = "/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr-v2/pytorch/default/1"
    # Model C: additional ByT5 checkpoint folder
    model_b_path: str = "/kaggle/input/datasets/zhuleyang0208/modelslarge"
    # Model D: additional ByT5 checkpoint folder
    model_d_path: str = "/kaggle/input/datasets/assiaben/final-byt5/byt5-akkadian-optimized-34x"

    # Logging controls
    log_pool_per_batch: bool = False
    log_tracebacks: bool = False

    max_input_length: int = 256
    max_new_tokens: int = 256
    batch_size: int = 8
    num_workers: int = 4
    num_buckets: int = 6

    num_beam_cands: int = 4
    num_beams: int = 8
    length_penalty: float = 1.3
    early_stopping: bool = True
    repetition_penalty: float = 1.2

    use_diverse_beam: bool = False
    num_diverse_cands: int = 4
    num_diverse_beams: int = 8
    num_beam_groups: int = 4
    diversity_penalty: float = 0.8

    use_sampling: bool = True
    sample_temperatures: List[float] = field(default_factory=lambda: [0.60, 0.80, 1.05])
    num_sample_per_temp: int = 2
    mbr_top_p: float = 0.92

    @property
    def num_sample_cands(self) -> int:
        return len(self.sample_temperatures) * self.num_sample_per_temp

    mbr_pool_cap: int = 32

    # MBR speed/robustness knobs
    # - per_model_pool_cap: cap candidates per model before merging (0 = no cap)
    # - mbr_min_words: filter candidates shorter than this (after postprocess)
    per_model_pool_cap: int = 24
    mbr_min_words: int = 3

    mbr_w_chrf: float = 0.55
    mbr_w_bleu: float = 0.25
    mbr_w_jaccard: float = 0.20
    mbr_w_length: float = 0.10

    use_mixed_precision: bool = True
    use_better_transformer: bool = True
    use_torch_compile: bool = True
    use_bucket_batching: bool = True
    use_adaptive_beams: bool = True
    aggressive_postprocessing: bool = True
    checkpoint_freq: int = 200

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        if not torch.cuda.is_available():
            self.use_mixed_precision = False
            self.use_better_transformer = False
            self.use_torch_compile = False

        self.use_amp = bool(self.use_mixed_precision and self.device.type == "cuda")
        self.amp_dtype = (
            "bf16" if _cuda_bf16_supported() else
            "fp16" if _fp16_supported(self.device) else
            "none"
        )

        assert self.num_beams >= self.num_beam_cands, "num_beams must be >= num_beam_cands"

        if self.use_diverse_beam:
            assert self.num_diverse_beams % self.num_beam_groups == 0
            assert self.num_diverse_beams >= self.num_diverse_cands


# 3. Logging utilities

def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "ensemble_mbr.log"),
        ],
    )
    return logging.getLogger("ensemble_mbr")


# 4. Preprocessing
#
# This block normalizes Akkadian transliteration before tokenization.
#
# Key steps:
#
# - convert ASCII-style diacritic notation into normalized Unicode,
# - normalize gap markers,
# - standardize determinatives,
# - normalize decimal/fraction expressions,
# - clean whitespace and symbol variants.
#
# This part is important because small notation mismatches can fragment the candidate space and hurt MBR consistency.

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú", "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
_GRAVE = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù", "A": "À", "E": "È", "I": "Ì", "U": "Ù"})


def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s


_ALLOWED_FRACS = [
    (1 / 6, "0.16666"), (1 / 4, "0.25"), (1 / 3, "0.33333"),
    (1 / 2, "0.5"), (2 / 3, "0.66666"), (3 / 4, "0.75"), (5 / 6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")


def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")


_WS_RE = re.compile(r"\s+")

_GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I
)


def _normalize_gaps_vec(ser: pd.Series) -> pd.Series:
    return ser.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)


_CHAR_TRANS = str.maketrans({
    "ḫ": "h", "Ḫ": "H", "ʾ": "",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "—": "-", "–": "-",
})
_SUB_X = "ₓ"

_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")

_PN_RE = re.compile(r"\bPN\b")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")

_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙",
    "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½",
}


def _frac_repl(m: re.Match) -> str:
    return _EXACT_FRAC_MAP[m.group(0)]


class OptimizedPreprocessor:
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        ser = pd.Series(texts).fillna("").astype(str)
        ser = ser.apply(_ascii_to_diacritics)
        ser = ser.str.replace(_DET_UPPER_RE, r"\1", regex=True)
        ser = ser.str.replace(_DET_LOWER_RE, r"{\1}", regex=True)
        ser = _normalize_gaps_vec(ser)
        ser = ser.str.translate(_CHAR_TRANS)
        ser = ser.str.replace(_SUB_X, "", regex=False)
        ser = ser.str.replace(_KUBABBAR_RE, "KÙ.BABBAR", regex=True)
        ser = ser.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        ser = ser.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()


# 5. Postprocessing
#
# This block is where a lot of the practical gain comes from.
#
# The idea is not to aggressively rewrite the model output, but to remove avoidable formatting noise so that MBR compares candidates on their *meaningful content* rather than superficial inconsistencies.
#
# I keep `/`, straight quotes, parentheses, and apostrophes when they may be meaningful.
#
# **Key corrections in this version:**
# - Parentheses `()` are **preserved** (they appear in the test set)
# - Curly quotes are **converted to straight quotes** (not deleted)
# - Commodity replacements only match space-prefixed hyphens
# - 5/12 shekel correctly maps to ⅓ shekel 15 grains
# - ḫ/Ḫ characters normalized to h/H

_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)", re.I
)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE = re.compile(r"\(\?\)")
# FIX #4: curly quotes → straight quotes (not deletion)
_CURLY_DQ_RE = re.compile("[\u201c\u201d]")  # " " → "
_CURLY_SQ_RE = re.compile("[\u2018\u2019]")  # ' ' → '
_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11,
              "XII": 12}
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")

# FIX #3: Removed '(' and ')' from forbidden chars — host confirmed
# parentheses appear in the test set, e.g. "(silver)", "(and)"
_FORBIDDEN_TRANS = str.maketrans("", "", '——<>⌈⌋⌊[]+ʾ;')

# FIX #2: Commodity regex now requires a SPACE before the hyphen,
# so "import-tax" and "kutānu-textiles" are NOT corrupted.
# Only standalone " -gold", " -tax", " -textiles" are replaced.
_COMMODITY_RE = re.compile(r'(?<=\s)-(gold|tax|textiles)\b')
_COMMODITY_REPL = {"gold": "pašallum gold", "tax": "šadduātum tax", "textiles": "kutānum textiles"}


def _commodity_repl(m: re.Match) -> str:
    return _COMMODITY_REPL[m.group(1)]


# FIX #1: 5/12 shekel → ⅓ shekel 15 grains (was ⅔, host confirmed)
_SHEKEL_REPLS = [
    (re.compile(r'5\s+11\s*/\s*12\s+shekels?', re.I), '6 shekels less 15 grains'),
    (re.compile(r'5\s*/\s*12\s+shekels?', re.I), '⅓ shekel 15 grains'),
    (re.compile(r'7\s*/\s*12\s+shekels?', re.I), '½ shekel 15 grains'),
    (re.compile(r'1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?', re.I), '15 grains'),
]

# FIX #6: Safer slash-alternative regex — require word chars around the slash
# and explicitly exclude digit/digit patterns (fractions like 1/3)
_SLASH_ALT_RE = re.compile(r'(?<![0-9/])\s+/\s+(?![0-9])\S+')

_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')
_MULTI_GAP_RE = re.compile(r'(?:<gap>\s*){2,}')

# FIX #7: Additional stray marks the host recommends removing
_EXTRA_STRAY_RE = re.compile(r'(?<!\w)(?:\.\.+|xx+)(?!\w)')

# FIX #5: ḫ/Ḫ normalization for translations (host confirmed not in test set)
_HACEK_TRANS = str.maketrans({"ḫ": "h", "Ḫ": "H"})


def _month_repl(m: re.Match) -> str:
    return f"Month {_ROMAN2INT.get(m.group(1).upper(), m.group(1))}"


class VectorizedPostprocessor:
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations).fillna("").astype(str)
        s = _normalize_gaps_vec(s)
        s = s.str.replace(_PN_RE, "<gap>", regex=True)
        s = s.str.replace(_COMMODITY_RE, _commodity_repl, regex=True)

        for pat, repl in _SHEKEL_REPLS:
            s = s.str.replace(pat, repl, regex=True)

        s = s.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        s = s.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
        s = s.str.replace(_SOFT_GRAM_RE, " ", regex=True)
        s = s.str.replace(_BARE_GRAM_RE, " ", regex=True)
        s = s.str.replace(_UNCERTAIN_RE, "", regex=True)
        s = s.str.replace(_STRAY_MARKS_RE, "", regex=True)
        s = s.str.replace(_EXTRA_STRAY_RE, "", regex=True)  # FIX #7
        s = s.str.replace(_SLASH_ALT_RE, "", regex=True)

        # FIX #4: curly → straight quotes (not deletion)
        s = s.str.replace(_CURLY_DQ_RE, '"', regex=True)
        s = s.str.replace(_CURLY_SQ_RE, "'", regex=True)

        s = s.str.replace(_MONTH_RE, _month_repl, regex=True)
        s = s.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)

        # Protect <gap> markers before stripping forbidden chars
        s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
        s = s.str.translate(_FORBIDDEN_TRANS)
        s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)

        # FIX #5: ḫ/Ḫ → h/H in translations
        s = s.str.translate(_HACEK_TRANS)

        s = s.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
        for n in range(4, 1, -1):
            pat = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
            s = s.str.replace(pat, r"\1", regex=True)

        s = s.str.replace(_PUNCT_SPACE_RE, r"\1", regex=True)
        s = s.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
        s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
        return s.tolist()


# 6. Dataset and bucket batching
#
# Bucketed batching reduces padding waste and keeps throughput more stable for variable-length transliterations.

class AkkadianDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor, logger: logging.Logger):
        self.sample_ids = df["id"].tolist()
        proc = preprocessor.preprocess_batch(df["transliteration"].tolist())
        self.input_texts = ["translate Akkadian to English: " + t for t in proc]
        logger.info(f"Dataset: {len(self.sample_ids)} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.sample_ids[idx], self.input_texts[idx]


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets, logger, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        lengths = [len(t.split()) for _, t in dataset]
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        bsize = max(1, len(sorted_idx) // max(1, num_buckets))
        self.buckets = [
            sorted_idx[i * bsize: None if i == num_buckets - 1 else (i + 1) * bsize]
            for i in range(num_buckets)
        ]

        for i, b in enumerate(self.buckets):
            if b:
                bl = [lengths[x] for x in b]
                logger.info(f"  Bucket {i}: {len(b)} samples, len [{min(bl)}, {max(bl)}]")

    def __iter__(self):
        for bucket in self.buckets:
            b = list(bucket)
            if self.shuffle:
                random.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                yield b[i:i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)


# 7. Model wrapper and candidate generation
#
# This is the main inference change compared with the earlier public version.
#
# Candidate sources
#
# For each model and each sample, the pool can include:
#
# - **standard beam candidates**
# - **diverse beam candidates** (optional)
# - **multi-temperature sampled candidates**
#
# In my current public LB 34.9 configuration:
#
# - diverse beam is **prepared but disabled**
# - multi-temperature sampling is **enabled**
# - the effective per-model pool is:
#
# `4 beam + 6 sampled = 10 candidates`
#
# Across two models, that gives about **20 candidates before deduplication**.
#
# Why this helped
#
# The previous version already benefited from cross-model pooling.
# This version improved further because the candidate pool became more heterogeneous, which gives MBR a better chance to select a consensus output that is both faithful and fluent.

class ModelWrapper:
    def __init__(self, model_path: str, cfg: EnsembleMBRConfig, logger: logging.Logger, label: str):
        self.cfg = cfg
        self.logger = logger
        self.label = label
        self.tokenizer = None
        self.model = None
        self._pool_logged = False

        logger.info(f"[{label}] Loading from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 多卡（双卡）支持：
        # - 如果有多块 GPU，则使用 transformers 的 device_map="auto" 自动把模型分片到多卡上；
        # - 否则保持原来的单卡行为（to(cfg.device)）。
        if cfg.device.type == "cuda" and torch.cuda.device_count() > 1:
            logger.info(f"[{label}] Detected {torch.cuda.device_count()} CUDA devices, using device_map='auto'")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                device_map="auto",
            ).eval()
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(cfg.device).eval()

        if cfg.device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if cfg.use_torch_compile and cfg.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
                logger.info(f"[{label}] torch.compile applied (reduce-overhead)")
            except Exception as e:
                logger.warning(f"[{label}] torch.compile skipped: {e}")

        n = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[{label}] {n:,} parameters")

        if cfg.device.type == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            logger.info(f"[{label}] GPU mem used: {used:.2f} GB")

        if cfg.use_better_transformer and cfg.device.type == "cuda":
            try:
                from optimum.bettertransformer import BetterTransformer
                self.model = BetterTransformer.transform(self.model)
                logger.info(f"[{label}] BetterTransformer applied")
            except Exception as e:
                logger.warning(f"[{label}] BetterTransformer skipped: {e}")

    def collate(self, batch_samples):
        ids = [s[0] for s in batch_samples]
        texts = [s[1] for s in batch_samples]
        enc = self.tokenizer(
            texts,
            max_length=self.cfg.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return ids, enc

    def generate_candidates(self, input_ids, attention_mask, beam_size: int) -> List[List[str]]:
        cfg = self.cfg
        B = input_ids.shape[0]
        ctx = _amp_ctx(cfg.device, cfg.use_amp)

        Rb = cfg.num_beam_cands
        Rd = cfg.num_diverse_cands if cfg.use_diverse_beam else 0
        Rs = cfg.num_sample_per_temp

        with ctx:
            nb = max(beam_size, Rb)
            beam_out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=nb,
                num_return_sequences=Rb,
                max_new_tokens=cfg.max_new_tokens,
                length_penalty=cfg.length_penalty,
                early_stopping=cfg.early_stopping,
                repetition_penalty=cfg.repetition_penalty,
                use_cache=True,
            )
            beam_texts = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)

            diverse_texts = []
            actual_Rd = 0
            if cfg.use_diverse_beam:
                try:
                    div_out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        num_beams=cfg.num_diverse_beams,
                        num_beam_groups=cfg.num_beam_groups,
                        diversity_penalty=cfg.diversity_penalty,
                        num_return_sequences=cfg.num_diverse_cands,
                        max_new_tokens=cfg.max_new_tokens,
                        length_penalty=cfg.length_penalty,
                        early_stopping=cfg.early_stopping,
                        repetition_penalty=cfg.repetition_penalty,
                        use_cache=True,
                    )
                    diverse_texts = self.tokenizer.batch_decode(div_out, skip_special_tokens=True)
                    actual_Rd = cfg.num_diverse_cands
                except Exception as e:
                    self.logger.warning(f"[{self.label}] Diverse beam failed ({e}), skipping")

            all_samp_texts = []
            num_temps = 0
            if cfg.use_sampling and cfg.sample_temperatures:
                num_temps = len(cfg.sample_temperatures)
                for temp in cfg.sample_temperatures:
                    try:
                        samp_out = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=True,
                            num_beams=1,
                            top_p=cfg.mbr_top_p,
                            temperature=temp,
                            num_return_sequences=Rs,
                            max_new_tokens=cfg.max_new_tokens,
                            repetition_penalty=cfg.repetition_penalty,
                            use_cache=True,
                        )
                        temp_texts = self.tokenizer.batch_decode(samp_out, skip_special_tokens=True)
                        all_samp_texts.extend(temp_texts)
                    except Exception as e:
                        self.logger.warning(
                            f"[{self.label}] Sampling @ temp={temp:.2f} failed ({e}), padding with empty strings"
                        )
                        all_samp_texts.extend([""] * (B * Rs))

        pools = []
        for i in range(B):
            pool = []
            pool.extend(beam_texts[i * Rb: (i + 1) * Rb])

            if diverse_texts and actual_Rd > 0:
                pool.extend(diverse_texts[i * actual_Rd: (i + 1) * actual_Rd])

            if all_samp_texts and num_temps > 0:
                for t_idx in range(num_temps):
                    seg_start = t_idx * B * Rs + i * Rs
                    pool.extend(all_samp_texts[seg_start: seg_start + Rs])

            pools.append(pool)

        if pools:
            if self.cfg.log_pool_per_batch:
                self.logger.info(
                    f"[{self.label}] Pool per sample: "
                    f"beam={Rb} + diverse={actual_Rd} + sample={num_temps}x{Rs}={num_temps * Rs} "
                    f"= {len(pools[0])} total"
                )
            elif not self._pool_logged:
                self.logger.info(
                    f"[{self.label}] Pool size: "
                    f"beam={Rb} + diverse={actual_Rd} + sample={num_temps}x{Rs}={num_temps * Rs} "
                    f"= {len(pools[0])} candidates/sample"
                )
                self._pool_logged = True

        return pools

    def unload(self):
        label = self.label
        try:
            from optimum.bettertransformer import BetterTransformer
            self.model = BetterTransformer.reverse(self.model)
        except Exception:
            pass

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            self.logger.info(f"[{label}] Unloaded. GPU free: {free:.2f} GB")


# 8. Weighted MBR selector
#
# This is the second major improvement.
#
# Earlier, I used **chrF++ only**.
# In this version, I use a weighted combination of:
#
# - **chrF++** for character-level robustness,
# - **BLEU** for word-level precision,
# - **Jaccard** for fast token overlap regularization,
# - **length bonus** to reduce pathological outputs.
#
# This is still a simple MBR implementation, but it is noticeably stronger in practice than relying on a single metric.

class MBRSelector:
    def __init__(
            self,
            pool_cap: int = 32,
            min_words: int = 0,
            w_chrf: float = 0.55,
            w_bleu: float = 0.25,
            w_jaccard: float = 0.20,
            w_length: float = 0.10,
    ):
        self._chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
        self._bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
        self.pool_cap = pool_cap
        self.min_words = int(min_words) if min_words else 0
        self.w_chrf = w_chrf
        self.w_bleu = w_bleu
        self.w_jaccard = w_jaccard
        self.w_length = w_length
        self._pw_total = max(w_chrf + w_bleu + w_jaccard, 1e-9)

    def _chrfpp(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(self._chrf_metric.sentence_score(a, [b]).score)

    def _bleu(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        try:
            return float(self._bleu_metric.sentence_score(a, [b]).score)
        except Exception:
            return 0.0

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta and not tb:
            return 100.0
        if not ta or not tb:
            return 0.0
        return 100.0 * len(ta & tb) / len(ta | tb)

    def _pairwise_score(self, a: str, b: str) -> float:
        s = (
                self.w_chrf * self._chrfpp(a, b)
                + self.w_bleu * self._bleu(a, b)
                + self.w_jaccard * self._jaccard(a, b)
        )
        return s / self._pw_total

    @staticmethod
    def _length_bonus(lengths: List[int], idx: int) -> float:
        if len(lengths) == 0:
            return 100.0
        median = float(np.median(lengths))
        sigma = max(median * 0.4, 5.0)
        z = (lengths[idx] - median) / sigma
        return 100.0 * math.exp(-0.5 * z * z)

    @staticmethod
    def _norm_key(x: str) -> str:
        """Normalization used only for dedup keys (keep original candidate text)."""
        if x is None:
            return ""
        s = str(x).strip().lower()
        if not s:
            return ""
        # unify gap marker variants
        s = re.sub(r"<\s*gap\s*>", "<gap>", s, flags=re.I)
        s = re.sub(r"\bpn\b", "<gap>", s)
        # collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        # remove trivial punctuation differences (keep digits/letters and <gap>)
        s = re.sub(r"(?!<gap>)[^\w\s<>/]+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _dedup(self, xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            raw = str(x).strip()
            if not raw:
                continue
            key = self._norm_key(raw)
            if not key or key in seen:
                continue
            out.append(raw)
            seen.add(key)
        return out

    def pick(self, candidates: List[str]) -> str:
        cands = self._dedup(candidates)
        if self.min_words:
            cands = [c for c in cands if len(c.split()) >= self.min_words]
        if self.pool_cap:
            cands = cands[:self.pool_cap]

        n = len(cands)
        if n == 0:
            return ""
        if n == 1:
            return cands[0]

        lengths = [len(c.split()) for c in cands]
        scores = []

        for i in range(n):
            pw = sum(
                self._pairwise_score(cands[i], cands[j])
                for j in range(n) if j != i
            ) / max(1, n - 1)

            lb = self._length_bonus(lengths, i)
            total = pw + self.w_length * lb
            scores.append(total)

        return cands[int(np.argmax(scores))]


# 9. End-to-end engine

class EnsembleMBREngine:
    def __init__(self, cfg: EnsembleMBRConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor()
        self.mbr = MBRSelector(
            pool_cap=cfg.mbr_pool_cap,
            min_words=cfg.mbr_min_words,
            w_chrf=cfg.mbr_w_chrf,
            w_bleu=cfg.mbr_w_bleu,
            w_jaccard=cfg.mbr_w_jaccard,
            w_length=cfg.mbr_w_length,
        )

    def _adaptive_beams(self, attn: torch.Tensor) -> int:
        if not self.cfg.use_adaptive_beams:
            return self.cfg.num_beams
        med = float(attn.sum(dim=1).float().median().item())
        short = max(self.cfg.num_beam_cands, self.cfg.num_beams // 2)
        return short if med < 100 else self.cfg.num_beams

    def _build_dataloader(self, dataset: AkkadianDataset, wrapper: ModelWrapper) -> DataLoader:
        if self.cfg.use_bucket_batching:
            sampler = BucketBatchSampler(dataset, self.cfg.batch_size, self.cfg.num_buckets, self.logger)
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.cfg.num_workers,
                collate_fn=wrapper.collate,
                pin_memory=(self.cfg.device.type == "cuda"),
                pin_memory_device=str(self.cfg.device) if self.cfg.device.type == "cuda" else "",
                persistent_workers=(self.cfg.num_workers > 0),
            )

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=wrapper.collate,
            pin_memory=(self.cfg.device.type == "cuda"),
            pin_memory_device=str(self.cfg.device) if self.cfg.device.type == "cuda" else "",
            persistent_workers=(self.cfg.num_workers > 0),
        )

    def _run_one_model(self, wrapper: ModelWrapper, dataset: AkkadianDataset) -> dict:
        dl = self._build_dataloader(dataset, wrapper)
        pools_by_id = {}

        with torch.inference_mode():
            for batch_ids, enc in tqdm(dl, desc=f"  [{wrapper.label}]"):
                input_ids = enc.input_ids.to(self.cfg.device, non_blocking=True)
                attn = enc.attention_mask.to(self.cfg.device, non_blocking=True)
                beam_size = self._adaptive_beams(attn)

                try:
                    batch_pools = wrapper.generate_candidates(input_ids, attn, beam_size)
                    for sid, pool in zip(batch_ids, batch_pools):
                        pools_by_id[str(sid)] = pool
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.error(f"OOM in [{wrapper.label}] — skipping batch")
                        if self.cfg.log_tracebacks:
                            self.logger.error(traceback.format_exc())
                        torch.cuda.empty_cache()
                        for sid in batch_ids:
                            pools_by_id.setdefault(str(sid), [])
                    else:
                        self.logger.error(f"[{wrapper.label}] RuntimeError: {e}")
                        if self.cfg.log_tracebacks:
                            self.logger.error(traceback.format_exc())
                        raise
                except Exception as e:
                    self.logger.error(f"[{wrapper.label}] batch error: {e}")
                    if self.cfg.log_tracebacks:
                        self.logger.error(traceback.format_exc())
                    for sid in batch_ids:
                        pools_by_id.setdefault(str(sid), [])

        return pools_by_id

    def run(self, test_df: pd.DataFrame) -> pd.DataFrame:
        cfg, logger = self.cfg, self.logger

        cands_per_sample = (
                cfg.num_beam_cands
                + (cfg.num_diverse_cands if cfg.use_diverse_beam else 0)
                + cfg.num_sample_cands
        )

        logger.info("=" * 60)
        logger.info("Ensemble × MBR  |  Cross-model candidate pooling  v4 (4 models)")
        logger.info(f"  Model A           : {cfg.model_a_path}")
        logger.info(f"  Model B           : {cfg.model_b_path}")
        logger.info(f"  Model C           : {cfg.model_c_path}")
        logger.info(f"  Model D           : {cfg.model_d_path}")
        logger.info(f"  Standard beam     : {cfg.num_beam_cands} (num_beams={cfg.num_beams})")
        logger.info(
            f"  Diverse beam      : {'ON' if cfg.use_diverse_beam else 'OFF'} "
            f"cands={cfg.num_diverse_cands}, groups={cfg.num_beam_groups}, penalty={cfg.diversity_penalty}"
        )
        logger.info(
            f"  Multi-temp sample : {'ON' if cfg.use_sampling else 'OFF'} "
            f"temps={cfg.sample_temperatures}, {cfg.num_sample_per_temp}/temp "
            f"→ {cfg.num_sample_cands} total"
        )
        logger.info(
            f"  Cands/model/sample: {cands_per_sample} "
            f"→ pool ≈ {cands_per_sample * 4} (4 models, pre-dedup)"
        )
        logger.info(f"  MBR pool cap      : {cfg.mbr_pool_cap}")
        logger.info(f"  AMP dtype         : {cfg.amp_dtype}")
        logger.info(f"  torch.compile     : {cfg.use_torch_compile}")
        logger.info(f"  batch_size        : {cfg.batch_size}")
        logger.info("=" * 60)

        dataset = AkkadianDataset(test_df, self.preprocessor, logger)
        sample_ids = [str(s) for s in dataset.sample_ids]

        logger.info("Phase 1/4 — Model A inference")
        wrapper_a = ModelWrapper(cfg.model_a_path, cfg, logger, "Model-A")
        pools_a = self._run_one_model(wrapper_a, dataset)
        wrapper_a.unload()
        del wrapper_a

        logger.info("Phase 2/4 — Model B inference")
        wrapper_b = ModelWrapper(cfg.model_b_path, cfg, logger, "Model-B")
        pools_b = self._run_one_model(wrapper_b, dataset)
        wrapper_b.unload()
        del wrapper_b

        logger.info("Phase 3/4 — Model C inference")
        wrapper_c = ModelWrapper(cfg.model_c_path, cfg, logger, "Model-C")
        pools_c = self._run_one_model(wrapper_c, dataset)
        wrapper_c.unload()
        del wrapper_c

        logger.info("Phase 4/4 — Model D inference")
        wrapper_d = ModelWrapper(cfg.model_d_path, cfg, logger, "Model-D")
        pools_d = self._run_one_model(wrapper_d, dataset)
        wrapper_d.unload()
        del wrapper_d

        logger.info("Phase 5/5 — Pool merge + MBR selection")
        results = []

        per_cap = int(cfg.per_model_pool_cap) if cfg.per_model_pool_cap else 0

        for sid in tqdm(sample_ids, desc="  MBR"):
            a = pools_a.get(sid, [])
            b = pools_b.get(sid, [])
            c = pools_c.get(sid, [])
            d = pools_d.get(sid, [])
            if per_cap > 0:
                a = a[:per_cap]
                b = b[:per_cap]
                c = c[:per_cap]
                d = d[:per_cap]

            combined = a + b + c + d
            pp = self.postprocessor.postprocess_batch(combined) if combined else []
            chosen = self.mbr.pick(pp)

            if not chosen or not chosen.strip():
                chosen = "The tablet is too damaged to translate."

            results.append((sid, chosen))

            if len(results) % cfg.checkpoint_freq == 0:
                ckpt = Path(cfg.output_dir) / f"checkpoint_{len(results)}.csv"
                pd.DataFrame(results, columns=["id", "translation"]).to_csv(ckpt, index=False)
                logger.info(f"  Checkpoint: {len(results)} rows → {ckpt}")

        result_df = pd.DataFrame(results, columns=["id", "translation"])
        self._validate(result_df)
        return result_df

    def _validate(self, df: pd.DataFrame):
        logger = self.logger
        logger.info("=" * 60)
        empty = df["translation"].str.strip().eq("").sum()
        lens = df["translation"].str.len()
        logger.info(f"Empty     : {empty} ({100 * empty / max(1, len(df)):.2f}%)")
        logger.info(f"Len mean  : {lens.mean():.1f}  median: {lens.median():.1f}  min: {lens.min()}  max: {lens.max()}")

        for idx in [0, len(df) // 4, len(df) // 2, 3 * len(df) // 4, len(df) - 1]:
            row = df.iloc[idx]
            logger.info(f"  ID {row['id']}: {str(row['translation'])[:80]}")

        logger.info("=" * 60)


# 10. Environment summary helper

def print_env(cfg: EnsembleMBRConfig):
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Mem  : {mem:.2f} GB")
        print(f"BF16     : {_cuda_bf16_supported()}")
    print(f"AMP dtype: {cfg.amp_dtype}")
    print(f"torch.compile: {cfg.use_torch_compile}")
    print()

    cands_per_sample = (
            cfg.num_beam_cands
            + (cfg.num_diverse_cands if cfg.use_diverse_beam else 0)
            + cfg.num_sample_cands
    )
    print(f"Candidates/sample/model : {cands_per_sample}")
    print(f"  ├─ standard beam      : {cfg.num_beam_cands} (num_beams={cfg.num_beams})")
    print(
        f"  ├─ diverse beam       : {cfg.num_diverse_cands if cfg.use_diverse_beam else 0} (groups={cfg.num_beam_groups}, penalty={cfg.diversity_penalty})")
    print(
        f"  └─ multi-temp sample  : {cfg.num_sample_cands} ({cfg.num_sample_per_temp}/temp × {cfg.sample_temperatures})")
    print(f"Total pool (4 models)   : ~{cands_per_sample * 4} (before dedup)")
    print()


# 11. Run inference and export submission
#
# This is the exact end-to-end submission path.

# ======================================================================
# Final notes
# ======================================================================

# ======================================================================
# Main execution
# ======================================================================

if __name__ == "__main__":
    cfg = EnsembleMBRConfig()
    logger = setup_logging(cfg.output_dir)

    print_env(cfg)

    logger.info(f"Loading test data: {cfg.test_data_path}")
    test_df = pd.read_csv(cfg.test_data_path, encoding="utf-8")
    logger.info(f"Test samples: {len(test_df)}")

    engine = EnsembleMBREngine(cfg, logger)
    results_df = engine.run(test_df)

    out_path = Path(cfg.output_dir) / "submission.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path}  ({len(results_df)} rows)")
    results_df.head()
    cfg_snap = {
        "model_a": cfg.model_a_path,
        "model_b": cfg.model_b_path,
        "model_c": cfg.model_c_path,
        "model_d": cfg.model_d_path,
        "num_beam_cands": cfg.num_beam_cands,
        "num_beams": cfg.num_beams,
        "length_penalty": cfg.length_penalty,
        "repetition_penalty": cfg.repetition_penalty,
        "use_diverse_beam": cfg.use_diverse_beam,
        "num_diverse_cands": cfg.num_diverse_cands,
        "num_diverse_beams": cfg.num_diverse_beams,
        "num_beam_groups": cfg.num_beam_groups,
        "diversity_penalty": cfg.diversity_penalty,
        "use_sampling": cfg.use_sampling,
        "sample_temperatures": cfg.sample_temperatures,
        "num_sample_per_temp": cfg.num_sample_per_temp,
        "num_sample_cands": cfg.num_sample_cands,
        "mbr_top_p": cfg.mbr_top_p,
        "mbr_w_chrf": cfg.mbr_w_chrf,
        "mbr_w_bleu": cfg.mbr_w_bleu,
        "mbr_w_jaccard": cfg.mbr_w_jaccard,
        "mbr_w_length": cfg.mbr_w_length,
        "mbr_pool_cap": cfg.mbr_pool_cap,
        "per_model_pool_cap": cfg.per_model_pool_cap,
        "mbr_min_words": cfg.mbr_min_words,
        "max_new_tokens": cfg.max_new_tokens,
        "amp_dtype": cfg.amp_dtype,
        "use_torch_compile": cfg.use_torch_compile,
        "batch_size": cfg.batch_size,
    }
    with open(Path(cfg.output_dir) / "ensemble_mbr_config.json", "w") as f:
        json.dump(cfg_snap, f, indent=2)

    print("Submission file:", out_path)
    print("Config saved to:", Path(cfg.output_dir) / "ensemble_mbr_config.json")