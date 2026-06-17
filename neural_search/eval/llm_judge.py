"""Optional LLM rubric judge for query-dataset relevance.

Requires: anthropic SDK + ANTHROPIC_API_KEY environment variable.
If either is missing, calling judge_pair() returns None and the caller
should skip gracefully.

Output schema (strict JSON):
{
  "label": 0-3,
  "confidence": 0.0-1.0,
  "rationale": "...",
  "supporting_evidence": ["..."],
  "missing_evidence": ["..."],
  "hard_negative_detected": false
}
"""
from __future__ import annotations

import json
import os
from typing import Any

import yaml

from neural_search.eval.evidence import PairEvidence


_JUDGMENT_SCHEMA = {
    "label": int,
    "confidence": float,
    "rationale": str,
    "supporting_evidence": list,
    "missing_evidence": list,
    "hard_negative_detected": bool,
}


def _default_judgment(rationale: str = "") -> dict:
    return {
        "label": 1,
        "confidence": 0.0,
        "rationale": rationale,
        "supporting_evidence": [],
        "missing_evidence": [],
        "hard_negative_detected": False,
    }


def load_config(config_path: str) -> dict | None:
    """Load judge config. Returns None if file not found."""
    try:
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        return None


def _build_prompt(pair: PairEvidence, rubric: str) -> str:
    q = pair.query
    d = pair.dataset
    return f"""## Research Query
Intent: {q.intent}
Query: {q.query_text}
Scientific goal: {q.scientific_goal}
Required modalities: {q.required_modalities}
Required species: {q.required_species}
Known failure modes (hard negatives): {q.hard_negatives}

## Dataset Evidence
Title: {d.title}
Description: {d.description or "Not provided"}
Species: {d.species}
Modalities: {d.modalities}
Brain regions: {d.regions}
Tasks: {d.tasks}
License: {d.license or "Unknown"}
Raw data available: {d.raw_data_available}
Data standards: {d.data_standards}
Metadata completeness: {d.metadata_completeness:.2f}

## Rubric
{rubric}

Respond with ONLY a JSON object matching this schema exactly:
{{"label": <0|1|2|3>, "confidence": <0.0-1.0>, "rationale": "<str>",
  "supporting_evidence": ["<str>", ...], "missing_evidence": ["<str>", ...],
  "hard_negative_detected": <true|false>}}"""


def _parse_judgment(text: str) -> dict | None:
    """Extract and validate the JSON judgment from model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    for key, expected_type in _JUDGMENT_SCHEMA.items():
        if key not in data:
            return None
        if not isinstance(data[key], expected_type):
            try:
                data[key] = expected_type(data[key])
            except (TypeError, ValueError):
                return None

    data["label"] = max(0, min(3, int(data["label"])))
    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    return data


def judge_pair(
    pair: PairEvidence,
    config: dict,
    *,
    n_runs: int | None = None,
) -> dict | None:
    """Judge a single pair. Returns a judgment dict or None if unavailable.

    Raises nothing — all errors are caught and None is returned.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic  # type: ignore[import]
    except ImportError:
        return None

    rubric = config.get("rubric", "")
    model = config.get("model", "claude-haiku-4-5-20251001")
    runs = n_runs or config.get("n_runs", 1)

    prompt = _build_prompt(pair, rubric)
    system = (
        "You are a scientific relevance judge. Output ONLY the requested JSON. "
        "Never add prose before or after the JSON object."
    )

    client = anthropic.Anthropic(api_key=api_key)
    judgments: list[dict] = []

    for _ in range(runs):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            parsed = _parse_judgment(text)
            if parsed:
                judgments.append(parsed)
        except Exception:
            continue

    if not judgments:
        return None

    if len(judgments) == 1:
        return judgments[0]

    from collections import Counter
    label = Counter(j["label"] for j in judgments).most_common(1)[0][0]
    confidence = sum(j["confidence"] for j in judgments) / len(judgments)
    representative = next(j for j in judgments if j["label"] == label)
    return {**representative, "label": label, "confidence": round(confidence, 4)}
