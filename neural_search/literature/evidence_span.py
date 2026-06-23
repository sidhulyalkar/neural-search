"""Locate a finding's supporting text span within its source abstract.

Extracted findings are LLM paraphrases of abstract text, so an exact
substring match is not always available. ``locate_evidence_span`` tries an
exact case-insensitive match first and falls back to the best matching block
found by difflib, so downstream evidence (GraphEvidence.char_start/char_end/
sentence_id) can point a reviewer at the right sentence even when the wording
was lightly paraphrased.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class EvidenceSpan:
    char_start: int
    char_end: int
    sentence_id: int
    match_method: str  # "exact" | "fuzzy"
    match_score: float


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets for each sentence in text."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _sentence_id_for_offset(sentence_spans: list[tuple[int, int]], offset: int) -> int:
    for idx, (start, end) in enumerate(sentence_spans):
        if start <= offset < end:
            return idx
    return max(len(sentence_spans) - 1, 0)


def locate_evidence_span(
    source_text: str,
    finding_text: str,
    *,
    min_score: float = 0.6,
) -> EvidenceSpan | None:
    """Find where ``finding_text`` is best supported within ``source_text``.

    Tries an exact case-insensitive substring match first; falls back to the
    longest matching block found by difflib's SequenceMatcher, scored as
    ``match_size / len(finding_text)``. Returns None if no match clears
    ``min_score`` (e.g. the finding was synthesized across sentences and has
    no contiguous textual anchor).
    """
    if not source_text or not finding_text:
        return None

    sentence_spans = _split_sentences(source_text)
    lowered_source = source_text.lower()
    lowered_finding = finding_text.lower().strip()
    if not lowered_finding:
        return None

    exact_idx = lowered_source.find(lowered_finding)
    if exact_idx != -1:
        char_end = exact_idx + len(lowered_finding)
        return EvidenceSpan(
            char_start=exact_idx,
            char_end=char_end,
            sentence_id=_sentence_id_for_offset(sentence_spans, exact_idx),
            match_method="exact",
            match_score=1.0,
        )

    matcher = difflib.SequenceMatcher(None, lowered_source, lowered_finding, autojunk=False)
    match = matcher.find_longest_match(0, len(lowered_source), 0, len(lowered_finding))
    if match.size == 0:
        return None

    score = match.size / len(lowered_finding)
    if score < min_score:
        return None

    char_start = match.a
    char_end = match.a + match.size
    return EvidenceSpan(
        char_start=char_start,
        char_end=char_end,
        sentence_id=_sentence_id_for_offset(sentence_spans, char_start),
        match_method="fuzzy",
        match_score=round(score, 3),
    )
