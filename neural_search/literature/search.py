"""Lightweight search over normalized literature shards and extracted findings."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal

TOKEN_RE = re.compile(r"[A-Za-z0-9_+-]+")


@dataclass(frozen=True)
class PaperResult:
    result_type: Literal["paper"] = "paper"
    paper_id: str = ""
    title: str = ""
    abstract_snippet: str | None = None
    year: int | None = None
    citation_count: int = 0
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    relevance_score: float = 0.0
    why_matched: list[str] = field(default_factory=list)
    linked_datasets: list[str] = field(default_factory=list)
    top_findings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FindingResult:
    result_type: Literal["finding"] = "finding"
    finding_id: str = ""
    finding_text: str = ""
    result_direction: str = ""
    regions: list[str] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    paper_id: str = ""
    paper_title: str = ""
    paper_year: int | None = None
    relevance_score: float = 0.0
    why_matched: list[str] = field(default_factory=list)


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for child in paths:
        if not child.exists():
            continue
        with child.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload


def _norm_set(values: Any) -> set[str]:
    if not values:
        return set()
    raw_values = values if isinstance(values, list | tuple | set) else [values]
    return {" ".join(_tokenize(str(value))) for value in raw_values if str(value).strip()}


def _matches_any(record_values: Any, required_values: Any) -> bool:
    required = _norm_set(required_values)
    if not required:
        return True
    available = _norm_set(record_values)
    return bool(available & required)


def _passes_paper_filters(record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    year = record.get("year")
    if filters.get("min_year") is not None and (year is None or year < filters["min_year"]):
        return False
    if filters.get("max_year") is not None and (year is None or year > filters["max_year"]):
        return False
    if filters.get("min_citations") is not None:
        if int(record.get("citation_count") or 0) < int(filters["min_citations"]):
            return False
    if filters.get("venue") and not _matches_any(record.get("venue"), filters["venue"]):
        return False
    if filters.get("topic") and not _matches_any(record.get("topics", []), filters["topic"]):
        return False
    return True


def _passes_finding_filters(record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    field_map = {
        "region": "regions",
        "regions": "regions",
        "species": "species",
        "modality": "modalities",
        "modalities": "modalities",
        "task": "tasks",
        "tasks": "tasks",
        "direction": "result_direction",
        "result_direction": "result_direction",
    }
    for filter_key, record_key in field_map.items():
        if filter_key in filters and not _matches_any(record.get(record_key), filters[filter_key]):
            return False
    return True


@lru_cache(maxsize=8)
def _load_index(path_str: str, text_fields: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    path = Path(path_str)
    records = list(_iter_jsonl(path))
    doc_freq: Counter[str] = Counter()
    for record in records:
        tokens: set[str] = set()
        for field_name in text_fields:
            value = record.get(field_name)
            if isinstance(value, list):
                tokens.update(_tokenize(" ".join(str(v) for v in value)))
            else:
                tokens.update(_tokenize(str(value or "")))
        doc_freq.update(tokens)

    n_docs = max(len(records), 1)
    idf = {
        token: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        for token, freq in doc_freq.items()
    }
    return records, idf


def _score_record(
    query_tokens: list[str],
    record: dict[str, Any],
    *,
    idf: dict[str, float],
    text_weights: dict[str, float],
) -> tuple[float, list[str]]:
    if not query_tokens:
        return 0.0, []

    score = 0.0
    matched_fields: list[str] = []
    query_counts = Counter(query_tokens)
    for field_name, weight in text_weights.items():
        value = record.get(field_name)
        text = " ".join(str(v) for v in value) if isinstance(value, list) else str(value or "")
        field_tokens = Counter(_tokenize(text))
        field_score = 0.0
        for token, qtf in query_counts.items():
            tf = field_tokens.get(token, 0)
            if tf:
                field_score += (1 + math.log(tf)) * idf.get(token, 0.1) * qtf
        if field_score > 0:
            score += field_score * weight
            matched_fields.append(field_name)
    return score, matched_fields


def _why(query_tokens: list[str], matched_fields: list[str]) -> list[str]:
    terms = ", ".join(sorted(set(query_tokens))[:6])
    return [f"matched {terms} in {field}" for field in matched_fields]


def search_papers(
    query: str,
    *,
    limit: int = 10,
    shard_dir: Path,
    filters: dict[str, Any] | None = None,
) -> list[PaperResult]:
    """Search OpenAlex paper shards using weighted lexical relevance."""

    query_tokens = _tokenize(query)
    if not query_tokens or not shard_dir.exists() or limit <= 0:
        return []

    records, idf = _load_index(str(shard_dir), ("title", "abstract", "venue", "topics"))
    scored: list[tuple[float, PaperResult]] = []
    for record in records:
        if not _passes_paper_filters(record, filters):
            continue
        score, matched_fields = _score_record(
            query_tokens,
            record,
            idf=idf,
            text_weights={"title": 3.0, "abstract": 1.0, "venue": 0.4, "topics": 0.8},
        )
        if score <= 0:
            continue
        abstract = record.get("abstract")
        snippet = abstract[:300] if isinstance(abstract, str) and abstract else None
        scored.append(
            (
                score,
                PaperResult(
                    paper_id=str(record.get("paper_id") or ""),
                    title=str(record.get("title") or ""),
                    abstract_snippet=snippet,
                    year=record.get("year"),
                    citation_count=int(record.get("citation_count") or 0),
                    venue=record.get("venue"),
                    doi=record.get("doi"),
                    url=record.get("url"),
                    relevance_score=round(score, 6),
                    why_matched=_why(query_tokens, matched_fields),
                    linked_datasets=list(record.get("linked_datasets") or []),
                    top_findings=list(record.get("top_findings") or [])[:2],
                ),
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in scored[:limit]]


def search_findings(
    query: str,
    *,
    limit: int = 10,
    findings_path: Path,
    filters: dict[str, Any] | None = None,
) -> list[FindingResult]:
    """Search extracted finding records, with optional concept filters."""

    query_tokens = _tokenize(query)
    if not query_tokens or not findings_path.exists() or limit <= 0:
        return []

    records, idf = _load_index(
        str(findings_path),
        ("finding_text", "regions", "species", "modalities", "tasks", "molecules", "cell_types"),
    )
    scored: list[tuple[float, FindingResult]] = []
    for record in records:
        if not _passes_finding_filters(record, filters):
            continue
        score, matched_fields = _score_record(
            query_tokens,
            record,
            idf=idf,
            text_weights={
                "finding_text": 3.0,
                "regions": 0.9,
                "species": 0.4,
                "modalities": 0.6,
                "tasks": 0.9,
                "molecules": 0.6,
                "cell_types": 0.6,
            },
        )
        if score <= 0:
            continue
        scored.append(
            (
                score,
                FindingResult(
                    finding_id=str(record.get("finding_id") or ""),
                    finding_text=str(record.get("finding_text") or ""),
                    result_direction=str(record.get("result_direction") or ""),
                    regions=list(record.get("regions") or []),
                    species=list(record.get("species") or []),
                    modalities=list(record.get("modalities") or []),
                    paper_id=str(record.get("paper_id") or ""),
                    paper_title=str(record.get("paper_title") or ""),
                    paper_year=record.get("paper_year"),
                    relevance_score=round(score, 6),
                    why_matched=_why(query_tokens, matched_fields),
                ),
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in scored[:limit]]
