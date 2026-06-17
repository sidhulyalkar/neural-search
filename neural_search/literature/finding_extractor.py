"""LLM-powered scientific finding extraction from paper abstracts."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_DIRECTIONS = {"increase", "decrease", "no_change", "correlation", "mechanism", "other"}

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FindingRecord:
    paper_id: str
    paper_doi: str | None
    finding_id: str
    finding_text: str
    result_direction: str
    regions: list[str]
    species: list[str]
    modalities: list[str]
    tasks: list[str]
    cell_types: list[str]
    molecules: list[str]
    confidence: float
    extraction_model: str
    extracted_at: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config. Return {} if file not found."""
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(title: str, abstract: str, config: dict[str, Any]) -> tuple[str, str]:
    """Return (system_prompt, user_message) from config template."""
    system_prompt: str = config.get("system_prompt", "")
    template: str = config.get("user_template", "Title: {title}\nAbstract: {abstract}")
    user_message = template.format(title=title, abstract=abstract)
    return system_prompt, user_message


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_direction(direction: str) -> str:
    return direction if direction in VALID_DIRECTIONS else "other"


def parse_findings(
    paper_id: str,
    paper_doi: str | None,
    response_text: str,
    model: str,
) -> list[FindingRecord]:
    """Parse JSON array from LLM response into FindingRecord list.

    Returns [] on parse failure — logs a warning, never raises.
    """
    try:
        raw: list[dict[str, Any]] = json.loads(response_text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("parse_findings: invalid JSON for paper %s", paper_id)
        return []

    if not isinstance(raw, list):
        logger.warning("parse_findings: expected list for paper %s", paper_id)
        return []

    extracted_at = datetime.now(tz=timezone.utc).isoformat()
    records: list[FindingRecord] = []

    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        confidence = _clamp(float(item.get("confidence", 0.0)))
        direction = _validate_direction(str(item.get("result_direction", "other")))
        records.append(
            FindingRecord(
                paper_id=paper_id,
                paper_doi=paper_doi,
                finding_id=f"{paper_id}:f{idx}",
                finding_text=str(item.get("finding_text", "")),
                result_direction=direction,
                regions=list(item.get("regions", [])),
                species=list(item.get("species", [])),
                modalities=list(item.get("modalities", [])),
                tasks=list(item.get("tasks", [])),
                cell_types=list(item.get("cell_types", [])),
                molecules=list(item.get("molecules", [])),
                confidence=confidence,
                extraction_model=model,
                extracted_at=extracted_at,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def extract_batch(
    papers: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    api_key: str | None = None,
) -> list[FindingRecord]:
    """Extract findings from a batch of papers using Claude Haiku.

    Returns [] if anthropic is not installed or no API key is available.
    """
    if anthropic is None:
        logger.warning("extract_batch: anthropic package not installed")
        return []

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        logger.warning("extract_batch: ANTHROPIC_API_KEY not set")
        return []

    model: str = config.get("model", "claude-haiku-4-5-20251001")
    max_tokens: int = config.get("max_tokens", 512)
    temperature: float = config.get("temperature", 0.0)
    client = anthropic.Anthropic(api_key=resolved_key)

    all_findings: list[FindingRecord] = []

    for paper in papers:
        abstract = paper.get("abstract", "")
        if not abstract:
            continue

        paper_id: str = paper.get("paper_id", "")
        paper_doi: str | None = paper.get("paper_doi")
        title: str = paper.get("title", "")

        system_prompt, user_message = build_prompt(title, abstract, config)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            response_text: str = response.content[0].text
        except Exception:
            logger.warning("extract_batch: API error for paper %s, skipping", paper_id)
            continue

        findings = parse_findings(paper_id, paper_doi, response_text, model)
        all_findings.extend(findings)

    return all_findings


# ---------------------------------------------------------------------------
# Corpus-level extraction
# ---------------------------------------------------------------------------


def _load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load set of already-processed paper IDs from checkpoint file."""
    if not checkpoint_path.exists():
        return set()
    try:
        data = json.loads(checkpoint_path.read_text())
        return set(data) if isinstance(data, list) else set()
    except (json.JSONDecodeError, OSError):
        return set()


def _save_checkpoint(checkpoint_path: Path, processed_ids: set[str]) -> None:
    checkpoint_path.write_text(json.dumps(sorted(processed_ids)))


def extract_findings_from_corpus(
    corpus_shards: list[Path],
    config: dict[str, Any],
    out_path: Path,
    *,
    checkpoint_path: Path | None = None,
    max_papers: int | None = None,
) -> int:
    """Process all shards, extract findings, write to out_path as JSONL.

    Returns total count of findings extracted.
    """
    if not corpus_shards:
        return 0

    processed_ids: set[str] = set()
    if checkpoint_path is not None:
        processed_ids = _load_checkpoint(checkpoint_path)

    batch_size: int = config.get("batch_size", 50)
    total_findings = 0
    papers_seen = 0
    batch: list[dict[str, Any]] = []

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as out_fh:
        for shard_path in corpus_shards:
            if not shard_path.exists():
                continue

            with shard_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        paper: dict[str, Any] = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    paper_id = paper.get("paper_id", "")
                    if paper_id in processed_ids:
                        continue

                    if max_papers is not None and papers_seen >= max_papers:
                        break

                    batch.append(paper)
                    papers_seen += 1

                    if len(batch) >= batch_size:
                        findings = extract_batch(batch, config)
                        for f in findings:
                            out_fh.write(json.dumps(f.__dict__) + "\n")
                            total_findings += 1
                            if checkpoint_path is not None:
                                processed_ids.add(f.paper_id)
                        batch = []

                        if papers_seen % 100 == 0:
                            logger.info("extract_findings_from_corpus: processed %d papers", papers_seen)

                else:
                    # shard exhausted without hitting max_papers; continue
                    continue
                break  # max_papers reached mid-shard

        # flush remaining batch
        if batch:
            findings = extract_batch(batch, config)
            for f in findings:
                out_fh.write(json.dumps(f.__dict__) + "\n")
                total_findings += 1
                if checkpoint_path is not None:
                    processed_ids.add(f.paper_id)

    if checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, processed_ids)

    return total_findings
