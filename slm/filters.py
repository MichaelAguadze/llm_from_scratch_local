"""The cleaning chain: pure functions, no I/O, no mutation.

Every filter is deterministic and side-effect free, so the whole chain is safe
to run inside a multiprocessing Pool and trivial to unit-test.

`clean_document` runs the chain cheapest-check-first and returns a CleanResult
carrying either the cleaned text or the reason it was dropped.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from . import config as C

# --------------------------------------------------------------- drop reasons --


class DropReason(str, Enum):
    KEPT = "kept"
    EMPTY = "empty"
    NO_LINES_LEFT = "no_lines_left"
    TOO_SHORT = "too_short"
    REPETITIVE = "repetitive"
    NON_ENGLISH = "non_english"
    OCR_GARBAGE = "ocr_garbage"


@dataclass(frozen=True)
class CleanResult:
    reason: DropReason
    text: str = ""

    @property
    def kept(self) -> bool:
        return self.reason is DropReason.KEPT


# ------------------------------------------------------------------- patterns --

_WHITESPACE = re.compile(r"\s+")
_ALNUM = re.compile(r"[a-zA-Z0-9]")
_WORD = re.compile(r"[A-Za-z]+")

# Whole lines matching any of these are boilerplate and get deleted.
_BOILERPLATE = re.compile(
    "|".join(
        (
            r"^\s*FORM\s+10-[KQ]\b",
            r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
            r"^\s*UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION\s*$",
            r"^\s*SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION\s*$",
            r"^\s*Washington,?\s+D\.?\s*C\.?\s+20549\s*$",
            r"^\s*/s/\s*.+$",
            r"^\s*TABLE\s+OF\s+CONTENTS\s*$",
            r"^\s*All\s+rights\s+reserved\.?\s*$",
            r"^\s*\(?[Cc]opyright\)?\s*(©|\(c\))?\s*\d{4}",
            r"^\s*©\s*\d{4}",
            r"^\s*\[?\s*(SIGNATURE|SIGNATURES)\s*\]?\s*$",
            r"^\s*Exhibit\s+\d+(\.\d+)?\s*$",
            r"^\s*See\s+accompanying\s+notes\b",
            r"^\s*<[A-Z/][^>]*>\s*$",
        )
    ),
    re.IGNORECASE,
)

# Heuristic OCR damage: mixed-case interior, stray digits inside words,
# or long consonant runs that real English words don't have.
_OCR_SUSPECT = re.compile(
    "|".join(
        (
            r"^[A-Za-z]*\d+[A-Za-z]+$",
            r"^[a-z]+[A-Z]{2,}",
            r"[bcdfghjklmnpqrstvwxz]{6,}",
            r"(.)\1{4,}",
        )
    )
)


# -------------------------------------------------------------------- filters --


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space."""
    return _WHITESPACE.sub(" ", text).strip()


def non_alnum_ratio(line: str) -> float:
    """Fraction of characters that are not letters or digits."""
    if not line:
        return 1.0
    alnum = len(_ALNUM.findall(line))
    return 1.0 - (alnum / len(line))


def filter_lines(text: str) -> str:
    """Keep only lines that are long enough and not mostly punctuation."""
    kept = [
        stripped
        for line in text.splitlines()
        if (stripped := line.strip())
        and len(stripped) >= C.MIN_LINE_CHARS
        and non_alnum_ratio(stripped) <= C.MAX_NON_ALNUM_RATIO
    ]
    return "\n".join(kept)


def strip_boilerplate(text: str) -> str:
    """Delete whole lines matching known boilerplate patterns."""
    return "\n".join(
        line for line in text.splitlines() if not _BOILERPLATE.match(line)
    )


def is_repetitive(text: str) -> bool:
    """True if the top-K n-grams dominate the document."""
    words = text.split()
    if len(words) < C.NGRAM_N * 2:
        return False
    grams = Counter(
        tuple(words[i : i + C.NGRAM_N])
        for i in range(len(words) - C.NGRAM_N + 1)
    )
    total = sum(grams.values())
    if total == 0:
        return False
    top = sum(count for _, count in grams.most_common(C.REPETITION_TOP_K))
    return (top / total) > C.MAX_REPETITION_RATIO


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isascii()) / len(text)


def is_english(text: str) -> bool:
    """langdetect on a prefix, with an ASCII-ratio fallback.

    langdetect is non-deterministic by default and raises on degenerate input,
    so the fallback carries the decision whenever it cannot answer.
    """
    prefix = text[: C.LANGDETECT_PREFIX_CHARS]
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = C.SEED
        return detect(prefix) == "en"
    except Exception:
        return ascii_ratio(prefix) >= C.MIN_ASCII_RATIO


def looks_like_ocr_garbage(text: str) -> bool:
    """True if too many words carry OCR damage signatures."""
    words = _WORD.findall(text[: C.LANGDETECT_PREFIX_CHARS]) or []
    tokens = text[: C.LANGDETECT_PREFIX_CHARS].split()
    if len(tokens) < 20:
        return False
    suspect = sum(1 for t in tokens if _OCR_SUSPECT.search(t))
    return (suspect / len(tokens)) > C.MAX_OCR_WORD_RATIO


# ---------------------------------------------------------------- the chain --


def clean_document(raw: str) -> CleanResult:
    """Run the full chain cheapest-check-first. First failure ends it."""
    if not raw or not raw.strip():
        return CleanResult(DropReason.EMPTY)

    text = strip_boilerplate(filter_lines(raw))
    if not text.strip():
        return CleanResult(DropReason.NO_LINES_LEFT)

    text = normalize_whitespace(text)
    if len(text) < C.MIN_DOC_CHARS:
        return CleanResult(DropReason.TOO_SHORT)

    if is_repetitive(text):
        return CleanResult(DropReason.REPETITIVE)

    if not is_english(text):
        return CleanResult(DropReason.NON_ENGLISH)

    if C.ENABLE_OCR_FILTER and looks_like_ocr_garbage(text):
        return CleanResult(DropReason.OCR_GARBAGE)

    return CleanResult(DropReason.KEPT, text)
