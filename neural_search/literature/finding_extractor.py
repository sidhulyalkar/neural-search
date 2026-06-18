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


@dataclass(frozen=True)
class LLMProviderConfig:
    name: str
    kind: str  # "anthropic" | "openai_compatible"
    api_key: str
    model: str
    base_url: str | None = None


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


def _repair_json(text: str) -> str:
    """Strip markdown fences and extract the first JSON array from LLM output.

    Local models frequently wrap JSON in ```json ... ``` blocks or add prose
    before/after the array. This function recovers the raw JSON in those cases.
    """
    text = text.strip()

    # Strip fenced code blocks: ```json ... ``` or ``` ... ```
    import re
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # If the text still doesn't start with '[', find the first '[' .. last ']'
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    return text


def parse_findings(
    paper_id: str,
    paper_doi: str | None,
    response_text: str,
    model: str,
) -> list[FindingRecord]:
    """Parse JSON array from LLM response into FindingRecord list.

    Returns [] on parse failure — logs a warning, never raises.
    Applies _repair_json() to handle markdown fences from local models.
    """
    cleaned = _repair_json(response_text)
    try:
        raw: list[dict[str, Any]] = json.loads(cleaned)
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


def extract_batch_openai_compatible(
    papers: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> list[FindingRecord]:
    """Extract findings using an OpenAI-compatible chat completions endpoint."""
    try:
        import openai
    except ImportError:  # pragma: no cover
        logger.warning("extract_batch_openai_compatible: openai package not installed")
        return []

    max_tokens: int = config.get("max_tokens", 512)
    temperature: float = config.get("temperature", 0.0)
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
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
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            response_text = response.choices[0].message.content or ""
        except Exception:
            logger.warning(
                "extract_batch_openai_compatible: API error for paper %s via %s, skipping",
                paper_id,
                model,
            )
            continue

        all_findings.extend(parse_findings(paper_id, paper_doi, response_text, model))

    return all_findings


def extract_batch_ollama(
    papers: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model: str,
    base_url: str = "http://localhost:11434",
) -> list[FindingRecord]:
    """Extract findings using the Ollama native /api/chat endpoint via requests.

    Handles JSON repair for models that wrap output in markdown fences.
    """
    try:
        import requests as req
    except ImportError:
        logger.warning("extract_batch_ollama: requests package not installed")
        return []

    max_tokens: int = config.get("max_tokens", 512)
    temperature: float = config.get("temperature", 0.0)
    chat_url = base_url.rstrip("/") + "/api/chat"
    all_findings: list[FindingRecord] = []

    for paper in papers:
        abstract = paper.get("abstract", "")
        if not abstract:
            continue

        paper_id: str = paper.get("paper_id", "")
        paper_doi: str | None = paper.get("paper_doi")
        title: str = paper.get("title", "")
        system_prompt, user_message = build_prompt(title, abstract, config)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = req.post(chat_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            response_text: str = data.get("message", {}).get("content", "")
        except Exception as exc:
            logger.warning("extract_batch_ollama: error for paper %s: %s", paper_id, exc)
            continue

        all_findings.extend(parse_findings(paper_id, paper_doi, response_text, model))

    return all_findings


def extract_batch_with_provider(
    papers: list[dict[str, Any]],
    config: dict[str, Any],
    provider: LLMProviderConfig,
) -> list[FindingRecord]:
    """Extract findings with a specific configured provider."""
    if provider.kind == "anthropic":
        provider_config = {**config, "model": provider.model}
        return extract_batch(papers, provider_config, api_key=provider.api_key)
    if provider.kind == "openai_compatible":
        return extract_batch_openai_compatible(
            papers,
            config,
            api_key=provider.api_key,
            base_url=provider.base_url or "",
            model=provider.model,
        )
    if provider.kind == "ollama":
        return extract_batch_ollama(
            papers,
            config,
            model=provider.model,
            base_url=provider.base_url or "http://localhost:11434",
        )
    logger.warning("Unknown LLM provider kind: %s", provider.kind)
    return []


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
    api_key: str | None = None,
    provider: LLMProviderConfig | None = None,
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
    # Append mode: preserve findings from previous runs when resuming
    open_mode = "a" if (checkpoint_path is not None and checkpoint_path.exists()) else "w"

    with out_path.open(open_mode) as out_fh:
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
                        batch_ids = {p.get("paper_id", "") for p in batch}
                        findings = (
                            extract_batch_with_provider(batch, config, provider)
                            if provider is not None
                            else extract_batch(batch, config, api_key=api_key)
                        )
                        for f in findings:
                            out_fh.write(json.dumps(f.__dict__) + "\n")
                            total_findings += 1
                        if checkpoint_path is not None:
                            processed_ids.update(batch_ids)
                            _save_checkpoint(checkpoint_path, processed_ids)
                        batch = []

                        if papers_seen % 100 == 0:
                            logger.info("extract_findings_from_corpus: processed %d papers", papers_seen)

                else:
                    # shard exhausted without hitting max_papers; continue
                    continue
                break  # max_papers reached mid-shard

        # flush remaining batch
        if batch:
            batch_ids = {p.get("paper_id", "") for p in batch}
            findings = (
                extract_batch_with_provider(batch, config, provider)
                if provider is not None
                else extract_batch(batch, config, api_key=api_key)
            )
            for f in findings:
                out_fh.write(json.dumps(f.__dict__) + "\n")
                total_findings += 1
            if checkpoint_path is not None:
                processed_ids.update(batch_ids)

    if checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, processed_ids)

    return total_findings
