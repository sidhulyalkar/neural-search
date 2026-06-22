"""Claims API — FastAPI router exposing synthesized claim objects to agents.

Loads claims from artifacts/claims/claims_validated.jsonl at first request
and caches in memory. Reset _claims_cache = None to force reload.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query

REPO_ROOT = Path(__file__).parent.parent.parent
CLAIMS_PATH = REPO_ROOT / "artifacts/claims/claims_validated.jsonl"

router = APIRouter()

_claims_cache: list[dict[str, Any]] | None = None


def _load_claims() -> list[dict[str, Any]]:
    global _claims_cache
    if _claims_cache is None:
        if not CLAIMS_PATH.exists():
            _claims_cache = []
        else:
            rows = []
            with CLAIMS_PATH.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            _claims_cache = rows
    return _claims_cache


def _agent_digest(claim: dict[str, Any]) -> str:
    """Generate a compact prior-loading sentence for an agent."""
    n = claim.get("n_supporting_findings", 0)
    n_contra = claim.get("n_contradicting_findings", 0)
    regions = ", ".join(claim.get("regions") or []) or "unspecified regions"
    direction = claim.get("direction", "correlates")
    mag = claim.get("magnitude_summary") or "N/A"
    contra_note = f" {n_contra} contradicting findings exist." if n_contra else ""
    return (
        f"{n} findings in {regions} show a {direction} effect "
        f"(magnitude: {mag}).{contra_note}"
    )


def _compact_claim(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_id": claim["claim_id"],
        "statement": claim.get("statement"),
        "confidence": claim.get("consensus_confidence"),
        "direction": claim.get("direction"),
        "regions": claim.get("regions", []),
        "species": claim.get("species", []),
        "n_evidence": claim.get("n_supporting_findings", 0),
        "status": claim.get("status", "active"),
        "contradicted_by": claim.get("contradicted_by", []),
        "supporting_datasets": claim.get("supporting_datasets", []),
        "agent_digest": claim.get("agent_digest") or _agent_digest(claim),
    }


@router.get("/api/claims")
def list_claims(
    regions: str | None = Query(None, description="Comma-separated region names"),
    species: str | None = Query(None),
    direction: str | None = Query(None),
    status: str | None = Query(None),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    claims = _load_claims()

    if regions:
        region_list = [r.strip().lower() for r in regions.split(",")]
        claims = [
            c for c in claims
            if any(r in [x.lower() for x in (c.get("regions") or [])] for r in region_list)
        ]
    if species:
        species_list = [s.strip().lower() for s in species.split(",")]
        claims = [
            c for c in claims
            if any(s in [x.lower() for x in (c.get("species") or [])] for s in species_list)
        ]
    if direction:
        claims = [c for c in claims if c.get("direction") == direction]
    if status:
        claims = [c for c in claims if c.get("status") == status]
    if min_confidence is not None:
        claims = [c for c in claims if (c.get("consensus_confidence") or 0) >= min_confidence]

    return {"claims": claims, "total": len(claims)}


@router.get("/api/claims/contradictions")
def list_contradictions() -> dict[str, Any]:
    claims = _load_claims()
    contested = [c for c in claims if c.get("status") == "contested"]
    return {
        "contested_claims": contested,
        "total": len(contested),
    }


@router.get("/api/claims/gaps")
def list_gaps(
    region: str | None = Query(None),
) -> dict[str, Any]:
    """Return (region, direction) combinations that have no claims."""
    claims = _load_claims()
    covered: set[tuple[str, str]] = set()
    for c in claims:
        for r in c.get("regions") or []:
            covered.add((r.lower(), c.get("direction", "other")))

    all_directions = {"increase", "decrease", "correlation", "no_change"}
    all_regions = {r.lower() for c in claims for r in (c.get("regions") or [])}

    if region:
        all_regions = {region.lower()}

    gaps = [
        {"region": r, "direction": d}
        for r in sorted(all_regions)
        for d in sorted(all_directions)
        if (r, d) not in covered
    ]
    return {"gaps": gaps, "total": len(gaps)}


@router.get("/api/claims/digest")
def digest(
    topic: str | None = Query(None, description="Filter by keyword in statement"),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """Compact claim objects for agent context loading."""
    claims = _load_claims()
    if topic:
        topic_lower = topic.lower()
        claims = [c for c in claims if topic_lower in (c.get("statement") or "").lower()]
    claims = claims[:limit]
    return {
        "claims": [_compact_claim(c) for c in claims],
        "total": len(claims),
        "generated_at": datetime.now(UTC).isoformat(),
    }


@router.get("/api/claims/{claim_id}/evidence")
def get_claim_evidence(claim_id: str) -> dict[str, Any]:
    """Evidence endpoint must be defined BEFORE the bare /{claim_id} route."""
    claim_id = unquote(claim_id)
    claims = _load_claims()
    for c in claims:
        if c.get("claim_id") == claim_id:
            return {
                "claim_id": claim_id,
                "supporting_datasets": c.get("supporting_datasets", []),
                "supporting_papers": c.get("supporting_papers", []),
                "contradicted_by": c.get("contradicted_by", []),
                "n_supporting_findings": c.get("n_supporting_findings", 0),
            }
    raise HTTPException(status_code=404, detail=f"Claim not found: {claim_id}")


@router.get("/api/claims/{claim_id}")
def get_claim(claim_id: str) -> dict[str, Any]:
    """Claim IDs contain colons (node:claim:*) but no slashes — {claim_id} without
    :path is sufficient and avoids routing conflicts with /evidence."""
    claim_id = unquote(claim_id)
    claims = _load_claims()
    for c in claims:
        if c.get("claim_id") == claim_id:
            return c
    raise HTTPException(status_code=404, detail=f"Claim not found: {claim_id}")
