"""Cluster findings into consensus claims and detect contradictions.

Three public functions:
  cluster_findings()       — group findings by (region, direction, species)
  synthesize_claim()       — LLM call to generate consensus claim text
  detect_contradictions()  — mark opposing claims as contested
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_OPPOSITE: dict[str, str] = {
    "increase": "decrease",
    "decrease": "increase",
}

DEFAULT_CONFIG = Path("configs/literature/synthesis_v1.yaml")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _opposite_directions(a: str, b: str) -> bool:
    return _OPPOSITE.get(a) == b


def _cluster_key(finding: dict[str, Any]) -> tuple[str, ...]:
    regions = tuple(sorted(finding.get("regions_normalized") or finding.get("regions") or []))
    direction = finding.get("result_direction", "other")
    species = tuple(sorted(finding.get("species") or []))
    return regions + (direction,) + species


def _claim_id_from_cluster(cluster: dict[str, Any]) -> str:
    key = json.dumps(
        {"regions": sorted(cluster["regions"]), "direction": cluster["direction"], "species": sorted(cluster["species"])},
        sort_keys=True,
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    slug = "_".join(cluster["regions"][:2] + [cluster["direction"]]).replace(" ", "_")[:40]
    return f"node:claim:{slug}_{digest}"


def cluster_findings(
    findings: list[dict[str, Any]],
    min_size: int = 3,
) -> list[dict[str, Any]]:
    """Group findings by (normalized_regions, direction, species).

    Returns clusters with >= min_size findings.
    Each cluster: {cluster_id, regions, direction, species, n_findings, findings}
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for f in findings:
        key = _cluster_key(f)
        buckets[key].append(f)

    clusters = []
    for key, group in buckets.items():
        if len(group) < min_size:
            continue
        regions = list(dict.fromkeys(
            r for f in group for r in (f.get("regions_normalized") or f.get("regions") or [])
        ))
        species = list(dict.fromkeys(s for f in group for s in (f.get("species") or [])))
        direction = group[0].get("result_direction", "other")
        cluster = {
            "regions": regions,
            "direction": direction,
            "species": species,
            "n_findings": len(group),
            "findings": group,
        }
        cluster["cluster_id"] = _claim_id_from_cluster(cluster)
        clusters.append(cluster)

    return clusters


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def synthesize_claim(
    cluster: dict[str, Any],
    client: Any,  # anthropic.Anthropic
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call Claude to generate a consensus claim for a finding cluster.

    Returns a claim dict ready for KG ingestion.
    Config keys: model, max_tokens, temperature, system_prompt, user_template, prompt_version.
    """
    if config is None:
        config = _load_config()

    sample = cluster["findings"][:10]
    findings_text = "\n".join(
        f"- {f.get('finding_text', '')[:200]}" for f in sample
    )
    user_message = config.get("user_template", "").format(
        regions=", ".join(cluster["regions"]),
        species=", ".join(cluster["species"]) or "unspecified",
        direction=cluster["direction"],
        n_findings=cluster["n_findings"],
        findings_text=findings_text,
    )

    response = client.messages.create(
        model=config.get("model", "claude-haiku-4-5-20251001"),
        max_tokens=config.get("max_tokens", 512),
        temperature=config.get("temperature", 0.0),
        system=config.get("system_prompt", ""),
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse synthesis response: %s", raw[:200])
        parsed = {
            "statement": raw[:500],
            "magnitude_summary": "N/A",
            "timescale": "unknown",
            "evidence_strength": "indirect",
        }

    paper_ids = list(dict.fromkeys(
        f.get("paper_id", "") for f in cluster["findings"] if f.get("paper_id")
    ))
    dataset_ids = list(dict.fromkeys(
        d for f in cluster["findings"] for d in (f.get("linked_datasets") or [])
    ))

    return {
        "claim_id": cluster["cluster_id"],
        "statement": parsed.get("statement", ""),
        "direction": cluster["direction"],
        "regions": cluster["regions"],
        "species": cluster["species"],
        "consensus_confidence": round(
            sum(f.get("confidence", 0.0) for f in cluster["findings"]) / cluster["n_findings"], 3
        ),
        "n_supporting_findings": cluster["n_findings"],
        "n_contradicting_findings": 0,
        "magnitude_summary": parsed.get("magnitude_summary", "N/A"),
        "timescale": parsed.get("timescale", "unknown"),
        "evidence_strength": parsed.get("evidence_strength", "indirect"),
        "status": "active",
        "supporting_papers": paper_ids[:20],
        "supporting_datasets": dataset_ids[:20],
        "contradicted_by": [],
        "synthesis_model": config.get("model", "claude-haiku-4-5-20251001"),
        "synthesis_prompt_version": config.get("prompt_version", "synthesis_v1"),
        "synthesized_at": _now(),
    }


def detect_contradictions(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find pairs of claims with opposing directions that share a brain region.

    Mutates and returns the claims list with `contradicted_by` and `status` updated.
    Status becomes "contested" when any contradiction is found.
    """
    claims = [dict(c) for c in claims]  # shallow copy each claim

    for i, a in enumerate(claims):
        for j, b in enumerate(claims):
            if i >= j:
                continue
            if not _opposite_directions(a["direction"], b["direction"]):
                continue
            shared_regions = set(a["regions"]) & set(b["regions"])
            if not shared_regions:
                continue
            if b["claim_id"] not in a["contradicted_by"]:
                a["contradicted_by"].append(b["claim_id"])
                a["status"] = "contested"
                a["n_contradicting_findings"] = a.get("n_contradicting_findings", 0) + b.get("n_supporting_findings", 1)
            if a["claim_id"] not in b["contradicted_by"]:
                b["contradicted_by"].append(a["claim_id"])
                b["status"] = "contested"
                b["n_contradicting_findings"] = b.get("n_contradicting_findings", 0) + a.get("n_supporting_findings", 1)

    return claims
