"""Shared heuristics for dataset reanalysis and method-transfer reports."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

METHOD_REGISTRY: list[dict[str, Any]] = [
    {
        "method": "behavior_aligned_embedding",
        "display_name": "Behavior-aligned embedding / CEBRA-style analysis",
        "novelty_year": 2023,
        "requires_any_modality": [
            "neuropixels",
            "extracellular",
            "ephys",
            "calcium",
            "fmri",
            "widefield",
        ],
        "requires_any_task": ["choice", "decision", "reward", "navigation", "motor", "behavior"],
        "requires": ["neural_data", "behavior"],
        "reinterpretation": "Align neural activity with behavior to expose latent state structure.",
    },
    {
        "method": "latent_dynamics_modeling",
        "display_name": "Latent neural dynamics modeling",
        "novelty_year": 2018,
        "requires_any_modality": ["neuropixels", "extracellular", "ephys", "calcium", "widefield"],
        "requires_any_task": ["choice", "decision", "reward", "motor", "navigation"],
        "requires": ["neural_data"],
        "reinterpretation": "Fit population dynamics to old recordings to test state-space hypotheses.",
    },
    {
        "method": "model_based_rl_reanalysis",
        "display_name": "Model-based RL reanalysis",
        "novelty_year": 2020,
        "requires_any_modality": ["fmri", "neuropixels", "extracellular", "ephys", "calcium"],
        "requires_any_task": ["reward", "reinforcement", "choice", "reversal", "go/nogo", "decision"],
        "requires": ["behavior"],
        "reinterpretation": "Re-estimate latent value, policy, and prediction-error variables.",
    },
    {
        "method": "cross_session_alignment",
        "display_name": "Cross-session representation alignment",
        "novelty_year": 2021,
        "requires_any_modality": ["neuropixels", "extracellular", "ephys", "calcium", "widefield"],
        "requires_any_task": ["choice", "decision", "reward", "motor", "navigation"],
        "requires": ["neural_data", "session_structure"],
        "reinterpretation": "Compare representational stability across sessions, animals, or cohorts.",
    },
    {
        "method": "multimodal_region_bridge",
        "display_name": "Multimodal same-region reinterpretation",
        "novelty_year": 2022,
        "requires_any_modality": ["fmri", "calcium", "neuropixels", "extracellular", "ephys", "widefield"],
        "requires_any_region": ["cortex", "hippocampus", "striatum", "amygdala", "thalamus", "ofc", "pfc"],
        "requires": ["neural_data"],
        "reinterpretation": "Use same-region datasets across modalities to connect scales of observation.",
    },
]

FIELD_GROUPS = {
    "modalities": ["modalities", "modality"],
    "species": ["species"],
    "tasks": ["tasks", "task", "behaviors", "behavioral_events"],
    "brain_regions": ["brain_regions", "regions"],
    "data_standards": ["data_standards", "standards"],
}


def stable_record_id(record: dict[str, Any]) -> str:
    source = str(record.get("source") or "unknown")
    source_id = str(record.get("source_id") or record.get("dataset_id") or record.get("id") or "unknown")
    if ":" in source_id and source_id.startswith(f"{source}:"):
        return source_id
    return f"{source}:{source_id}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _coerce_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("label", "id", "name", "value"):
            if value.get(key):
                return str(value[key])
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                return _coerce_label(parsed)
        return text
    return str(value)


def values_for(record: dict[str, Any], field_group: str) -> list[str]:
    values: list[str] = []
    for field in FIELD_GROUPS.get(field_group, [field_group]):
        raw = record.get(field)
        if raw is None:
            continue
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            label = _coerce_label(item)
            if label:
                values.append(label)
    return sorted(dict.fromkeys(values), key=str.casefold)


def norm_text(values: list[str]) -> str:
    return " ".join(values).casefold().replace("_", " ").replace("-", " ")


def has_any(text: str, needles: list[str]) -> bool:
    return any(needle.casefold() in text for needle in needles)


def infer_year(record: dict[str, Any]) -> int | None:
    text = " ".join(
        str(record.get(field) or "")
        for field in ["created_at", "updated_at", "date", "published_at", "description", "title"]
    )
    matches = [int(match) for match in re.findall(r"\b(19[8-9]\d|20[0-2]\d)\b", text)]
    return min(matches) if matches else None


def usability_flags(record: dict[str, Any]) -> dict[str, bool]:
    raw = record.get("usability_flags") or {}
    if not isinstance(raw, dict):
        raw = {}
    text = " ".join(
        [
            str(record.get("title") or ""),
            str(record.get("description") or ""),
            norm_text(values_for(record, "tasks")),
            norm_text(values_for(record, "modalities")),
            norm_text(values_for(record, "data_standards")),
        ]
    ).casefold()
    modalities_text = norm_text(values_for(record, "modalities"))
    standards_text = norm_text(values_for(record, "data_standards"))
    return {
        "neural_data": bool(
            raw.get("has_neural_data")
            or has_any(modalities_text, ["fmri", "ephys", "neuropixels", "calcium", "widefield", "eeg"])
        ),
        "behavior": bool(
            raw.get("has_behavior")
            or has_any(text, ["choice", "reward", "task", "behavior", "trial", "navigation", "motor"])
        ),
        "raw_data": bool(raw.get("has_raw_data") or has_any(text, ["raw", "nwb", "bids"])),
        "session_structure": bool(has_any(text, ["session", "subject", "cohort", "animal", "participant"])),
        "standard_format": bool(has_any(standards_text, ["nwb", "bids"])),
    }


def score_method(record: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    modalities = norm_text(values_for(record, "modalities"))
    tasks = norm_text(values_for(record, "tasks"))
    regions = norm_text(values_for(record, "brain_regions"))
    flags = usability_flags(record)
    missing: list[str] = []
    score = 0.0

    modality_match = has_any(modalities, method.get("requires_any_modality", []))
    task_match = has_any(tasks, method.get("requires_any_task", []))
    region_match = has_any(regions, method.get("requires_any_region", []))

    if method.get("requires_any_modality"):
        score += 0.3 if modality_match else 0.0
        if not modality_match:
            missing.append("modality")
    if method.get("requires_any_task"):
        score += 0.25 if task_match else 0.0
        if not task_match:
            missing.append("task")
    if method.get("requires_any_region"):
        score += 0.2 if region_match else 0.0
        if not region_match:
            missing.append("brain_region")

    required_flags = method.get("requires", [])
    if required_flags:
        per_flag = 0.25 / len(required_flags)
        for flag in required_flags:
            if flags.get(flag):
                score += per_flag
            else:
                missing.append(flag)

    if flags["standard_format"]:
        score += 0.1
    if flags["raw_data"]:
        score += 0.1

    return {
        "method": method["method"],
        "display_name": method["display_name"],
        "score": round(min(score, 1.0), 3),
        "missing_requirements": sorted(dict.fromkeys(missing)),
        "reinterpretation": method["reinterpretation"],
        "novelty_year": method["novelty_year"],
    }


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    method_scores = [score_method(record, method) for method in METHOD_REGISTRY]
    method_scores.sort(key=lambda row: (-row["score"], row["method"]))
    missing_metadata = [
        field
        for field in ["modalities", "species", "tasks", "brain_regions", "data_standards"]
        if not values_for(record, field)
    ]
    flags = usability_flags(record)
    if not flags["raw_data"]:
        missing_metadata.append("raw_data")
    if not flags["behavior"]:
        missing_metadata.append("behavior")
    return {
        "record_id": stable_record_id(record),
        "title": str(record.get("title") or stable_record_id(record)),
        "source": str(record.get("source") or "unknown"),
        "year": infer_year(record),
        "modalities": values_for(record, "modalities"),
        "species": values_for(record, "species"),
        "tasks": values_for(record, "tasks"),
        "brain_regions": values_for(record, "brain_regions"),
        "data_standards": values_for(record, "data_standards"),
        "missing_metadata": sorted(dict.fromkeys(missing_metadata)),
        "top_methods": method_scores[:3],
        "best_method_score": method_scores[0]["score"] if method_scores else 0.0,
    }

