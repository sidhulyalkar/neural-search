"""Small JSONL persistence helpers for field-state artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from neural_search.field_state.schemas import (
    BenchmarkGap,
    FieldClaim,
    FieldOpportunity,
)

T = TypeVar("T", bound=BaseModel)

ARTIFACT_DIR = Path("artifacts/field_state")
CLAIMS_PATH = ARTIFACT_DIR / "claims.jsonl"
BENCHMARK_GAPS_PATH = ARTIFACT_DIR / "benchmark_gaps.jsonl"
OPPORTUNITIES_PATH = ARTIFACT_DIR / "opportunities.jsonl"
QRELS_CANDIDATES_PATH = ARTIFACT_DIR / "qrels_candidates.jsonl"
QRELS_REVIEWS_PATH = ARTIFACT_DIR / "qrels_reviews.jsonl"
ADJUDICATED_QRELS_PATH = ARTIFACT_DIR / "adjudicated_qrels.jsonl"
QRELS_AGREEMENT_PATH = ARTIFACT_DIR / "qrels_agreement.json"
EVAL_SNAPSHOT_PATH = ARTIFACT_DIR / "eval_snapshot.json"
CLAIM_EVIDENCE_SUGGESTIONS_PATH = ARTIFACT_DIR / "claim_evidence_suggestions.jsonl"

REPORT_DIR = Path("reports/field_state")
WEAK_CLAIMS_REPORT = REPORT_DIR / "weak_claims.md"
BENCHMARK_GAPS_REPORT = REPORT_DIR / "benchmark_gaps.md"
TOP_OPPORTUNITIES_REPORT = REPORT_DIR / "top_opportunities.md"
LATEST_SNAPSHOT_REPORT = REPORT_DIR / "latest_snapshot.md"
QRELS_AGREEMENT_REPORT = REPORT_DIR / "qrels_agreement.md"
EVAL_SNAPSHOT_REPORT = REPORT_DIR / "eval_snapshot.md"
CLAIM_EVIDENCE_UPDATE_REPORT = REPORT_DIR / "claim_evidence_update.md"
WHITEPAPER_VALIDATION_REPORT = REPORT_DIR / "whitepaper_validation_report.md"


def resolve_path(path: Path, root: Path | None = None) -> Path:
    """Resolve a repo-relative path against the supplied root."""
    if path.is_absolute():
        return path
    return (root or Path.cwd()) / path


def write_jsonl(path: Path, rows: Sequence[BaseModel], root: Path | None = None) -> Path:
    """Write Pydantic rows to a JSONL file."""
    output_path = resolve_path(path, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(row.model_dump_json() for row in rows)
    output_path.write_text(f"{text}\n" if text else "", encoding="utf-8")
    return output_path


def read_jsonl(
    path: Path,
    model_type: type[T],
    root: Path | None = None,
) -> list[T]:
    """Read Pydantic rows from a JSONL file."""
    input_path = resolve_path(path, root)
    if not input_path.exists():
        return []
    rows: list[T] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model_type.model_validate_json(line))
    return rows


def write_field_state(
    claims: list[FieldClaim],
    gaps: list[BenchmarkGap],
    opportunities: list[FieldOpportunity],
    root: Path | None = None,
) -> dict[str, Path]:
    """Write all field-state JSONL artifacts."""
    return {
        "claims": write_jsonl(CLAIMS_PATH, claims, root),
        "benchmark_gaps": write_jsonl(BENCHMARK_GAPS_PATH, gaps, root),
        "opportunities": write_jsonl(OPPORTUNITIES_PATH, opportunities, root),
    }


def read_claims(root: Path | None = None) -> list[FieldClaim]:
    """Read stored field claims."""
    return read_jsonl(CLAIMS_PATH, FieldClaim, root)


def read_benchmark_gaps(root: Path | None = None) -> list[BenchmarkGap]:
    """Read stored benchmark gaps."""
    return read_jsonl(BENCHMARK_GAPS_PATH, BenchmarkGap, root)


def read_opportunities(root: Path | None = None) -> list[FieldOpportunity]:
    """Read stored opportunities."""
    return read_jsonl(OPPORTUNITIES_PATH, FieldOpportunity, root)
