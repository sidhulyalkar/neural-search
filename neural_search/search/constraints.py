"""Hard-negative query constraint parsing and dataset filtering."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from neural_search.analysis_affordances import AFFORDANCE_IDS
from neural_search.ontology import get_ontology, normalize_text
from neural_search.scientific_labels import RULES

NEGATIVE_FIELDS = (
    "hard_excluded_modalities",
    "hard_excluded_tasks",
    "hard_excluded_sources",
    "hard_excluded_species",
    "hard_excluded_regions",
    "hard_excluded_dataset_types",
    "hard_excluded_analysis_affordances",
    "hard_excluded_recording_devices",
)

NEGATION_RE = re.compile(
    r"\b(?:exclude|excluding|without|but\s+not|not|no|NOT)\b\s+(?P<span>[^.;]+)",
    flags=re.IGNORECASE,
)

STOP_RE = re.compile(
    r"\b(?:with|where|that|which|and enough|suitable for|datasets?|studies|recordings?)\b",
    flags=re.IGNORECASE,
)

BEHAVIOR_ONLY_ALIASES = (
    "behavior only",
    "behavior-only",
    "behaviour only",
    "behaviour-only",
    "pure behavior",
    "pure behavior-only",
    "pure behavior only",
)

SOURCE_ALIASES = {
    "dandi": ("dandi", "dandiset"),
    "openneuro": ("openneuro", "open neuro"),
    "openalex": ("openalex", "open alex"),
    "demo": ("demo",),
}

RECORDING_DEVICE_ALIASES = {
    "utah_array": ("utah array", "utah arrays", "utah_array"),
}

EXTRA_MODALITY_ALIASES = {
    "extracellular_ephys": ("extracellular ephys", "electrophysiology", "ephys", "spikes"),
    "calcium_imaging": ("calcium imaging", "two photon", "two-photon", "2p"),
    "fiber_photometry": ("fiber photometry", "photometry"),
    "behavior_video": ("behavior video", "video"),
    "pose_tracking": ("pose tracking", "deeplabcut", "sleap"),
    "ieeg": ("ieeg", "intracranial eeg", "seeg"),
    "ecog": ("ecog", "electrocorticography"),
    "eeg": ("eeg", "scalp eeg"),
    "fmri": ("fmri", "functional mri", "bold"),
    "utah_array": ("utah array", "utah arrays", "utah_array"),
}

EXTRA_TASK_ALIASES = {
    "auditory_processing": ("auditory", "auditory tasks", "auditory task"),
    "seizure_monitoring": ("seizure", "seizure monitoring", "seizure datasets"),
    "sleep": ("sleep", "sleep studies", "sleep datasets"),
    "virtual_reality": ("virtual reality", "vr"),
    "head_fixed": ("head fixed", "head-fixed"),
}

EXTRA_ANALYSIS_ALIASES = {
    "q_learning_modeling": ("q learning", "q-learning", "q learning model"),
    "event_aligned_activity": ("event aligned", "event alignment", "psth"),
    "choice_decoding": ("choice decoding", "decode choice"),
    "seizure_detection": ("seizure detection",),
}


def _empty_constraints() -> dict[str, list[str]]:
    return {field: [] for field in NEGATIVE_FIELDS}


def _add_alias(vocab: dict[str, dict[str, set[str]]], field: str, canonical: str, alias: str) -> None:
    canonical_norm = (
        str(canonical).strip().casefold().replace("-", "_").replace(" ", "_")
    )
    alias_norm = normalize_text(alias)
    if canonical_norm and alias_norm:
        vocab.setdefault(field, {}).setdefault(canonical_norm, set()).add(alias_norm)


def _rule_field(label_type: str) -> str | None:
    return {
        "modality": "hard_excluded_modalities",
        "task": "hard_excluded_tasks",
        "species": "hard_excluded_species",
        "brain_region": "hard_excluded_regions",
        "analysis_goal": "hard_excluded_analysis_affordances",
    }.get(label_type)


def _vocabulary(config: Mapping[str, Any] | None = None) -> dict[str, dict[str, set[str]]]:
    vocab: dict[str, dict[str, set[str]]] = {}
    for rule in RULES:
        field = _rule_field(rule.label_type)
        if field is None:
            continue
        _add_alias(vocab, field, rule.label_id, rule.label_id)
        _add_alias(vocab, field, rule.label_id, rule.label)
        for term in rule.terms:
            _add_alias(vocab, field, rule.label_id, term)

    try:
        for task in get_ontology().tasks:
            _add_alias(vocab, "hard_excluded_tasks", task.id, task.id)
            _add_alias(vocab, "hard_excluded_tasks", task.id, task.label)
            for synonym in task.synonyms:
                _add_alias(vocab, "hard_excluded_tasks", task.id, synonym)
    except Exception:
        pass

    for canonical, aliases in EXTRA_MODALITY_ALIASES.items():
        for alias in aliases:
            _add_alias(vocab, "hard_excluded_modalities", canonical, alias)
    for canonical, aliases in EXTRA_TASK_ALIASES.items():
        for alias in aliases:
            _add_alias(vocab, "hard_excluded_tasks", canonical, alias)
    for canonical, aliases in SOURCE_ALIASES.items():
        for alias in aliases:
            _add_alias(vocab, "hard_excluded_sources", canonical, alias)
    for canonical, aliases in RECORDING_DEVICE_ALIASES.items():
        for alias in aliases:
            _add_alias(vocab, "hard_excluded_recording_devices", canonical, alias)
            _add_alias(vocab, "hard_excluded_modalities", canonical, alias)
    for canonical, aliases in EXTRA_ANALYSIS_ALIASES.items():
        for alias in aliases:
            _add_alias(vocab, "hard_excluded_analysis_affordances", canonical, alias)
    for analysis_id in AFFORDANCE_IDS:
        _add_alias(vocab, "hard_excluded_analysis_affordances", analysis_id, analysis_id)
        _add_alias(
            vocab,
            "hard_excluded_analysis_affordances",
            analysis_id,
            analysis_id.replace("_", " "),
        )
    if config:
        for canonical, aliases in (config.get("species_aliases") or {}).items():
            for alias in aliases:
                _add_alias(vocab, "hard_excluded_species", str(canonical), str(alias))
        for canonical, aliases in (config.get("analysis_intents") or {}).items():
            for alias in aliases:
                _add_alias(
                    vocab,
                    "hard_excluded_analysis_affordances",
                    str(canonical),
                    str(alias),
                )
    for alias in BEHAVIOR_ONLY_ALIASES:
        _add_alias(vocab, "hard_excluded_dataset_types", "behavior_only", alias)
    return vocab


def _negative_spans(query: str) -> list[str]:
    spans: list[str] = []
    for match in NEGATION_RE.finditer(query):
        span = match.group("span").strip(" ,")
        next_negation = NEGATION_RE.search(span, 1)
        if next_negation:
            span = span[: next_negation.start()].strip(" ,")
        stop = STOP_RE.search(span)
        if stop and stop.start() > 0:
            span = span[: stop.start()].strip(" ,")
        if span:
            spans.append(span)
    return spans


def _contains_alias(span: str, alias: str) -> bool:
    if not alias:
        return False
    return re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", span) is not None


def parse_hard_negative_constraints(
    query: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Parse explicit hard-negative constraints from a natural-language query."""

    constraints = _empty_constraints()
    spans = [normalize_text(span) for span in _negative_spans(query)]
    if not spans:
        return constraints

    vocab = _vocabulary(config)
    for span in spans:
        for field, aliases_by_id in vocab.items():
            for canonical, aliases in aliases_by_id.items():
                if any(_contains_alias(span, alias) for alias in aliases):
                    constraints[field].append(canonical)

    return {
        field: sorted(dict.fromkeys(values))
        for field, values in constraints.items()
    }


def _values(obj: Any, name: str) -> list[str]:
    if isinstance(obj, Mapping):
        value = obj.get(name, [])
    else:
        value = getattr(obj, name, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]


def _card_labels(card: Mapping[str, Any] | None, group: str) -> list[str]:
    if not card:
        return []
    labels = card.get("scientific_labels", {}) if isinstance(card, Mapping) else {}
    values = labels.get(group, []) if isinstance(labels, Mapping) else []
    output: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            output.extend(str(value.get(key, "")) for key in ("id", "label"))
        else:
            output.append(str(value))
    return output


def _norm_set(values: Sequence[str]) -> set[str]:
    return {normalize_text(value) for value in values if normalize_text(value)}


def _dataset_analysis_values(dataset: Any, card: Mapping[str, Any] | None) -> set[str]:
    values = _values(dataset, "analysis_affordances")
    normalized: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            normalized.append(str(value.get("analysis_id", "")))
        else:
            normalized.append(str(value))
    normalized.extend(_values(card or {}, "suggested_analyses"))
    return _norm_set(normalized)


def _is_behavior_only(dataset: Any) -> bool:
    modalities = _norm_set(_values(dataset, "modalities"))
    has_behavior = bool(getattr(dataset, "has_behavior", False))
    if isinstance(dataset, Mapping):
        has_behavior = bool(dataset.get("has_behavior", has_behavior))
    behavior_modalities = _norm_set([
        "behavior_video",
        "behavior_tracking",
        "pose_tracking",
        "deeplabcut",
        "sleap",
        "facemap",
        "pupil_tracking",
    ])
    neural_modalities = _norm_set([
        "calcium_imaging",
        "extracellular_ephys",
        "electrophysiology",
        "neuropixels",
        "spikes",
        "lfp",
        "eeg",
        "ecog",
        "ieeg",
        "seeg",
        "fmri",
        "fiber_photometry",
        "utah_array",
    ])
    return has_behavior and bool(modalities) and not bool(modalities & neural_modalities) and bool(
        modalities & behavior_modalities
    )


def negative_constraint_violations(
    dataset: Any,
    card: Mapping[str, Any] | None,
    constraints: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return hard-negative constraints violated by a dataset/card pair."""

    modalities = _norm_set([*_values(dataset, "modalities"), *_card_labels(card, "modalities")])
    tasks = _norm_set([*_values(dataset, "tasks"), *_card_labels(card, "tasks")])
    species = _norm_set([*_values(dataset, "species"), *_card_labels(card, "species")])
    regions = _norm_set([*_values(dataset, "brain_regions"), *_card_labels(card, "brain_regions")])
    behaviors = _norm_set([*_values(dataset, "behaviors"), *_card_labels(card, "behaviors")])
    sources = _norm_set(
        [
            *_values(dataset, "source"),
            *_values(dataset, "data_standards"),
            *_values(dataset, "source_id"),
        ]
    )
    analyses = _dataset_analysis_values(dataset, card)
    text = normalize_text(
        " ".join(
            [
                *_values(dataset, "title"),
                *_values(dataset, "description"),
                *_values(dataset, "source"),
                *_values(dataset, "source_id"),
            ]
        )
    )

    checks = {
        "hard_excluded_modalities": modalities,
        "hard_excluded_tasks": tasks | behaviors,
        "hard_excluded_species": species,
        "hard_excluded_sources": sources,
        "hard_excluded_regions": regions,
        "hard_excluded_analysis_affordances": analyses,
        "hard_excluded_recording_devices": modalities,
    }

    violations: list[str] = []
    for field, actual_values in checks.items():
        excluded = _norm_set(list(constraints.get(field, [])))
        matches = excluded & actual_values
        if field == "hard_excluded_recording_devices":
            matches |= {value for value in excluded if value and value in text}
        violations.extend(sorted(matches))

    excluded_dataset_types = set(constraints.get("hard_excluded_dataset_types", [])) | _norm_set(
        list(constraints.get("hard_excluded_dataset_types", []))
    )
    if {"behavior_only", "behavior only"} & excluded_dataset_types:
        if _is_behavior_only(dataset):
            violations.append("behavior_only")
    return sorted(dict.fromkeys(violations))
