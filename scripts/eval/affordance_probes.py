"""Affordance probes for silver qrels generation.

Each probe tests whether a dataset record appears to support a specific
analysis workflow (spike sorting, calcium imaging analysis, RL model fitting,
etc.).  Probes return a structured result indicating whether the affordance
is supported, with confidence, required evidence present/absent, and rationale.

Probes do NOT infer unsupported affordances from vague descriptions alone.
Confidence is penalised when critical fields are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scripts.eval.silver_qrels_schema import LabelingFunctionVote

# ---------------------------------------------------------------------------
# Probe result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AffordanceProbeResult:
    """Output of a single affordance probe for one dataset record."""

    affordance: str
    supported: bool | None  # True / False / None (unknown)
    confidence: float  # 0.0–1.0
    required_evidence_present: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    import re
    return re.sub(r"[_\-]", " ", s.lower()).strip()


def _has_any(text: str, terms: list[str]) -> list[str]:
    nt = _norm(text)
    return [t for t in terms if _norm(t) in nt]


def _field_text(record: dict[str, Any], *keys: str) -> str:
    """Concatenate string values from given record keys."""
    parts = []
    for key in keys:
        val = record.get(key)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            parts.extend(str(v) for v in val)
    return " ".join(parts)


def _modalities(record: dict[str, Any]) -> list[str]:
    raw = record.get("modalities") or []
    return [_norm(str(m)) for m in raw]


def _tasks(record: dict[str, Any]) -> list[str]:
    raw = (record.get("tasks") or []) + (record.get("behaviors") or [])
    return [_norm(str(t)) for t in raw]


def _standards(record: dict[str, Any]) -> list[str]:
    raw = record.get("data_standards") or []
    return [_norm(str(s)) for s in raw]


# ---------------------------------------------------------------------------
# Probe 1 — Spike sorting
# ---------------------------------------------------------------------------


def probe_spike_sorting(record: dict[str, Any]) -> AffordanceProbeResult:
    """Dataset supports spike sorting or single-unit analysis."""
    mods = _modalities(record)
    text = _norm(_field_text(record, "title", "description"))
    present: list[str] = []
    missing: list[str] = []

    ephys_terms = ["ephys", "extracellular", "neuropixels", "tetrode", "silicon probe", "single unit", "multi unit"]
    raw_terms = ["raw", "ap band", "continuous", "waveform"]

    ephys_hits = [t for t in ephys_terms if any(t in m for m in mods) or t in text]
    if ephys_hits:
        present.append(f"electrophysiology indicators: {ephys_hits}")
    else:
        missing.append("electrophysiology or neuropixels modality")

    raw_hits = [t for t in raw_terms if t in text]
    if raw_hits:
        present.append(f"raw data indicators: {raw_hits}")
    else:
        missing.append("raw ephys / AP band indicators")

    if not ephys_hits:
        return AffordanceProbeResult(
            affordance="spike_sorting",
            supported=False,
            confidence=0.80,
            missing_evidence=missing,
            rationale="No electrophysiology modality detected; spike sorting not applicable.",
        )

    confidence = 0.60 + 0.15 * bool(raw_hits) + 0.10 * bool(
        record.get("has_raw_data")
    )
    return AffordanceProbeResult(
        affordance="spike_sorting",
        supported=True,
        confidence=min(confidence, 0.90),
        required_evidence_present=present,
        missing_evidence=missing,
        rationale="Electrophysiology modality present; spike sorting likely applicable.",
    )


# ---------------------------------------------------------------------------
# Probe 2 — Calcium imaging analysis
# ---------------------------------------------------------------------------


def probe_calcium_imaging(record: dict[str, Any]) -> AffordanceProbeResult:
    mods = _modalities(record)
    text = _norm(_field_text(record, "title", "description"))
    present: list[str] = []
    missing: list[str] = []

    ca_terms = ["calcium", "two photon", "2p", "gcamp", "fluorescence", "widefield", "miniscope"]
    ca_hits = [t for t in ca_terms if any(t in m for m in mods) or t in text]

    roi_terms = ["roi", "region of interest", "traces", "df/f", "deltaf", "suite2p", "deconvolution"]
    roi_hits = [t for t in roi_terms if t in text]

    if ca_hits:
        present.append(f"calcium imaging indicators: {ca_hits}")
    else:
        missing.append("calcium imaging modality")

    if roi_hits:
        present.append(f"trace/ROI indicators: {roi_hits}")
    else:
        missing.append("dF/F traces or ROI indicators")

    if not ca_hits:
        return AffordanceProbeResult(
            affordance="calcium_imaging_analysis",
            supported=False,
            confidence=0.75,
            missing_evidence=missing,
            rationale="No calcium imaging modality detected.",
        )

    confidence = 0.60 + 0.15 * bool(roi_hits)
    return AffordanceProbeResult(
        affordance="calcium_imaging_analysis",
        supported=True,
        confidence=min(confidence, 0.85),
        required_evidence_present=present,
        missing_evidence=missing,
        rationale="Calcium imaging modality present; analysis applicable.",
    )


# ---------------------------------------------------------------------------
# Probe 3 — Q-learning / RL model fitting
# ---------------------------------------------------------------------------


def probe_rl_model_fitting(record: dict[str, Any]) -> AffordanceProbeResult:
    text = _norm(_field_text(record, "title", "description"))
    tasks = _tasks(record)
    task_text = " ".join(tasks)
    full_text = text + " " + task_text
    present: list[str] = []
    missing: list[str] = []

    choice_terms = ["choice", "choices", "decision", "reward", "outcome", "trial"]
    rl_terms = ["q learning", "reinforcement learning", "bandit", "operant"]
    struct_terms = ["trial level", "trial-level", "trial by trial", "events"]

    choice_hits = [t for t in choice_terms if t in full_text]
    rl_hits = [t for t in rl_terms if t in full_text]
    struct_hits = [t for t in struct_terms if t in full_text]

    if choice_hits:
        present.append(f"choice/reward indicators: {choice_hits}")
    else:
        missing.append("trial-level choices or rewards")

    if struct_hits:
        present.append(f"trial structure indicators: {struct_hits}")
    else:
        missing.append("trial-level event structure")

    if not choice_hits:
        return AffordanceProbeResult(
            affordance="rl_model_fitting",
            supported=False,
            confidence=0.70,
            missing_evidence=missing,
            rationale="No choice/reward/trial evidence; RL model fitting not supported.",
        )

    confidence = 0.55 + 0.15 * bool(struct_hits) + 0.10 * bool(rl_hits)
    return AffordanceProbeResult(
        affordance="rl_model_fitting",
        supported=True,
        confidence=min(confidence, 0.85),
        required_evidence_present=present,
        missing_evidence=missing,
        rationale="Choice/reward indicators present; RL model fitting plausible.",
    )


# ---------------------------------------------------------------------------
# Probe 4 — Pose analysis
# ---------------------------------------------------------------------------


def probe_pose_analysis(record: dict[str, Any]) -> AffordanceProbeResult:
    mods = _modalities(record)
    text = _norm(_field_text(record, "title", "description"))
    present: list[str] = []
    missing: list[str] = []

    video_terms = ["video", "camera", "behavioural video", "behavior video"]
    pose_terms = ["pose", "deeplabcut", "dlc", "facemap", "sleap", "keypoint", "tracking"]

    video_hits = [t for t in video_terms if any(t in m for m in mods) or t in text]
    pose_hits = [t for t in pose_terms if t in text]

    if video_hits:
        present.append(f"video indicators: {video_hits}")
    else:
        missing.append("video or camera modality")

    if pose_hits:
        present.append(f"pose tracking indicators: {pose_hits}")
    else:
        missing.append("pose tracking / DLC / SLEAP indicators")

    if not video_hits and not pose_hits:
        return AffordanceProbeResult(
            affordance="pose_analysis",
            supported=False,
            confidence=0.75,
            missing_evidence=missing,
            rationale="No video or pose-tracking evidence found.",
        )

    confidence = 0.55 + 0.20 * bool(pose_hits) + 0.10 * bool(video_hits)
    return AffordanceProbeResult(
        affordance="pose_analysis",
        supported=True if (video_hits or pose_hits) else None,
        confidence=min(confidence, 0.85),
        required_evidence_present=present,
        missing_evidence=missing,
        rationale="Video or pose-tracking indicators present.",
    )


# ---------------------------------------------------------------------------
# Probe 5 — Neural decoding
# ---------------------------------------------------------------------------


def probe_neural_decoding(record: dict[str, Any]) -> AffordanceProbeResult:
    mods = _modalities(record)
    text = _norm(_field_text(record, "title", "description"))
    present: list[str] = []
    missing: list[str] = []

    neural_terms = ["ephys", "calcium", "neuropixels", "eeg", "ecog", "fmri", "bold",
                    "single unit", "lfp", "multi unit"]
    behav_terms = ["trial", "label", "stimulus", "category", "condition", "class", "behavior", "behaviour"]

    neural_hits = [t for t in neural_terms if any(t in m for m in mods) or t in text]
    behav_hits = [t for t in behav_terms if t in text]

    if neural_hits:
        present.append(f"neural modality: {neural_hits}")
    else:
        missing.append("neural modality (ephys, calcium, EEG, fMRI)")

    if behav_hits:
        present.append(f"behavioural labels/trials: {behav_hits}")
    else:
        missing.append("behavioural labels / trial structure")

    if not neural_hits:
        return AffordanceProbeResult(
            affordance="neural_decoding",
            supported=False,
            confidence=0.75,
            missing_evidence=missing,
            rationale="No neural modality; decoding not applicable.",
        )

    confidence = 0.55 + 0.20 * bool(behav_hits)
    return AffordanceProbeResult(
        affordance="neural_decoding",
        supported=True if behav_hits else None,
        confidence=min(confidence, 0.85),
        required_evidence_present=present,
        missing_evidence=missing,
        rationale="Neural modality present; decoding plausible if behavioural labels exist.",
    )


# ---------------------------------------------------------------------------
# Probe 6 — Cross-dataset comparison
# ---------------------------------------------------------------------------


def probe_cross_dataset_comparison(record: dict[str, Any]) -> AffordanceProbeResult:
    present: list[str] = []
    missing: list[str] = []

    species = record.get("species") or []
    mods = record.get("modalities") or []
    tasks = record.get("tasks") or []
    n_sessions = record.get("metadata_json", {}).get("asset_summary", {}).get("sessions")

    if species:
        present.append(f"species: {species}")
    else:
        missing.append("species metadata")

    if mods:
        present.append(f"modalities: {mods}")
    else:
        missing.append("modality metadata")

    if tasks:
        present.append(f"tasks: {tasks}")
    else:
        missing.append("task metadata")

    if n_sessions:
        present.append(f"session count: {n_sessions}")
    else:
        missing.append("session/subject count")

    completeness = len(present) / (len(present) + len(missing)) if (present or missing) else 0.0
    confidence = 0.40 + 0.40 * completeness

    supported: bool | None = completeness >= 0.5
    return AffordanceProbeResult(
        affordance="cross_dataset_comparison",
        supported=supported,
        confidence=confidence,
        required_evidence_present=present,
        missing_evidence=missing,
        rationale=(
            f"Metadata completeness {completeness:.0%}; "
            "cross-dataset comparison requires species/modality/task alignment."
        ),
    )


# ---------------------------------------------------------------------------
# Probe 7 — Meta-analysis
# ---------------------------------------------------------------------------


def probe_meta_analysis(record: dict[str, Any]) -> AffordanceProbeResult:
    present: list[str] = []
    missing: list[str] = []

    for field_name in ("species", "modalities", "tasks", "brain_regions"):
        val = record.get(field_name) or []
        if val:
            present.append(f"{field_name}: {val}")
        else:
            missing.append(field_name)

    license_val = str(record.get("license", "") or "")
    if license_val and license_val.lower() not in ("unknown", "none", ""):
        present.append(f"license: {license_val}")
    else:
        missing.append("license")

    completeness = len(present) / (len(present) + len(missing)) if (present or missing) else 0.0
    confidence = 0.35 + 0.45 * completeness

    return AffordanceProbeResult(
        affordance="meta_analysis",
        supported=completeness >= 0.60,
        confidence=confidence,
        required_evidence_present=present,
        missing_evidence=missing,
        rationale=(
            f"Structured metadata completeness {completeness:.0%}; "
            "meta-analysis requires complete species/modality/task/license."
        ),
    )


# ---------------------------------------------------------------------------
# Registry and conversion
# ---------------------------------------------------------------------------

ALL_PROBES = [
    probe_spike_sorting,
    probe_calcium_imaging,
    probe_rl_model_fitting,
    probe_pose_analysis,
    probe_neural_decoding,
    probe_cross_dataset_comparison,
    probe_meta_analysis,
]

_AFFORDANCE_TO_VOTE: dict[bool | None, int | None] = {
    True: 2,
    None: None,
    False: 0,
}


def probe_to_vote(result: AffordanceProbeResult) -> LabelingFunctionVote:
    """Convert an AffordanceProbeResult to a LabelingFunctionVote."""
    vote = _AFFORDANCE_TO_VOTE.get(result.supported)
    evidence = result.required_evidence_present + [f"missing: {m}" for m in result.missing_evidence]
    return LabelingFunctionVote(
        source="affordance_probe",
        vote=vote,
        confidence=result.confidence,
        rationale=f"{result.affordance}: {result.rationale}",
        evidence=evidence,
    )


def apply_affordance_probes(
    record: dict[str, Any],
) -> list[tuple[AffordanceProbeResult, LabelingFunctionVote]]:
    """Run all affordance probes and return (result, vote) pairs."""
    return [(p(record), probe_to_vote(p(record))) for p in ALL_PROBES]
