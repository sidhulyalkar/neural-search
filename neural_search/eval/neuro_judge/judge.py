"""Multi-backend LLM judge for neuro_judge.

Backends:
  MockNeuroJudge     — deterministic, no network (for tests)
  AnthropicNeuroJudge — Anthropic API (requires ANTHROPIC_API_KEY + anthropic pkg)
  OpenAINeuroJudge   — OpenAI-compatible API (requires OPENAI_API_KEY + openai pkg)
  GeminiNeuroJudge   — Google Gemini API (requires GEMINI_API_KEY)
  LocalHFNeuroJudge  — local HuggingFace causal LM (requires transformers + torch)
  BrainGPTAdapter    — optional BrainGPT over Mistral-7B (skips if unavailable)

build_neuro_judge(backend, **kwargs) returns a judge implementing NeuroJudgeProtocol.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

from neural_search.eval.neuro_judge.evidence_packet import (
    PROMPT_VERSION_DEFAULT,
    EvidencePacket,
    NeuroJudgment,
)
from neural_search.eval.neuro_judge.prompt import PROMPT_VERSION, build_judge_prompt

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class NeuroJudgeProtocol(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def judge(self, packet: EvidencePacket) -> NeuroJudgment: ...


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------


class JudgeParseError(ValueError):
    """Raised when the model returns malformed JSON or an out-of-range field."""


_DIMENSIONS = ("species", "modality", "brain_region", "task", "affordance", "raw_data")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _normalise_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _normalise_values(values: list[str]) -> set[str]:
    return {_normalise_token(str(value)) for value in values if str(value).strip()}


def _infer_expected_from_text(text: str, dimension: str) -> set[str]:
    t = text.lower()
    inferred: set[str] = set()
    if dimension == "species":
        if "human" in t:
            inferred.add("human")
        if any(term in t for term in ("mouse", "mice", "murine")):
            inferred.add("mouse")
        if "rat" in t or "rodent" in t:
            inferred.update({"rat", "mouse"} if "rodent" in t else {"rat"})
        if any(term in t for term in ("macaque", "monkey", "primate")):
            inferred.add("macaque")
    elif dimension == "modality":
        if any(term in t for term in ("fmri", "bold", "mri")):
            inferred.add("fmri")
        if any(term in t for term in ("extracellular", "single unit", "single-unit", "spike", "neuropixels", "lfp")):
            inferred.add("extracellular_ephys")
        if any(term in t for term in ("calcium", "two-photon", "2-photon", "gcamp")):
            inferred.add("calcium_imaging")
        if "eeg" in t:
            inferred.add("eeg")
    elif dimension == "brain_region":
        region_terms = {
            "hippocampus": ("hippocampus", "ca1", "ca3", "place cell"),
            "lip": ("lip", "lateral intraparietal"),
            "pfc": ("pfc", "prefrontal"),
            "mt_v5": ("mt/v5", "middle temporal", "mt area", "v5"),
            "visual_cortex": ("visual cortex", "v1", "orientation selectivity"),
            "striatum": ("striatum", "ventral striatum"),
            "default_mode_network": ("default mode", "dmn"),
        }
        for canonical, terms in region_terms.items():
            if any(term in t for term in terms):
                inferred.add(canonical)
    elif dimension == "task":
        task_terms = {
            "spatial_navigation": ("spatial navigation", "open field", "place cell"),
            "random_dot_motion": ("random dot", "saccade", "decision"),
            "working_memory": ("working memory", "n-back", "n back"),
            "reward_learning": ("reward", "reinforcement learning", "q-learning", "prediction error"),
            "visual_stimulation": ("visual", "orientation", "grating", "object recognition"),
            "sleep": ("sleep", "spindle", "slow oscillation"),
            "reaching": ("reaching", "motor planning"),
            "resting_state": ("resting-state", "resting state"),
        }
        for canonical, terms in task_terms.items():
            if any(term in t for term in terms):
                inferred.add(canonical)
    elif dimension == "affordance":
        if any(term in t for term in ("spike sorting", "kilosort", "single unit", "single-unit")):
            inferred.add("spike_sorting")
        if any(term in t for term in ("raw ap", "ap-band", "ap band", "raw electrophysiology")):
            inferred.add("raw_ap_band")
        if any(term in t for term in ("model fitting", "q-learning", "trial-level")):
            inferred.add("model_fitting")
    return inferred


def _canonical_modality(values: set[str]) -> set[str]:
    canonical: set[str] = set()
    for value in values:
        v = value.lower()
        if any(term in v for term in ("fmri", "bold", "mri")):
            canonical.add("fmri")
        if any(term in v for term in ("extracellular", "ephys", "neuropixels", "single_unit", "single-unit", "lfp", "spike")):
            canonical.add("extracellular_ephys")
        if any(term in v for term in ("calcium", "two_photon", "two-photon", "gcamp")):
            canonical.add("calcium_imaging")
        if "eeg" in v:
            canonical.add("eeg")
    return canonical or values


def _canonical_species(values: set[str]) -> set[str]:
    canonical: set[str] = set()
    for value in values:
        v = value.lower()
        if "human" in v:
            canonical.add("human")
        if any(term in v for term in ("mouse", "mice", "murine")):
            canonical.add("mouse")
        if "rat" in v:
            canonical.add("rat")
        if "rodent" in v:
            canonical.update({"mouse", "rat"})
        if any(term in v for term in ("macaque", "monkey", "primate")):
            canonical.add("macaque")
    return canonical or values


def _canonical_regions(values: set[str]) -> set[str]:
    canonical: set[str] = set()
    for value in values:
        v = value.lower()
        if "any" == v:
            canonical.add("any")
        if any(term in v for term in ("hippocampus", "ca1", "ca3")):
            canonical.add("hippocampus")
        if v == "lip" or "lateral_intraparietal" in v:
            canonical.add("lip")
        if any(term in v for term in ("pfc", "prefrontal")):
            canonical.add("pfc")
        if any(term in v for term in ("mt", "v5", "middle_temporal")):
            canonical.add("mt_v5")
        if any(term in v for term in ("visual_cortex", "v1", "visual")):
            canonical.add("visual_cortex")
        if "striatum" in v:
            canonical.add("striatum")
        if any(term in v for term in ("default_mode", "dmn")):
            canonical.add("default_mode_network")
        if "entorhinal" in v:
            canonical.add("entorhinal_cortex")
    return canonical or values


def _canonical_tasks(values: set[str]) -> set[str]:
    canonical: set[str] = set()
    for value in values:
        v = value.lower()
        if "any" == v:
            canonical.add("any")
        if any(term in v for term in ("spatial", "open_field", "navigation", "maze")):
            canonical.add("spatial_navigation")
        if any(term in v for term in ("random_dot", "saccade", "decision")):
            canonical.add("random_dot_motion")
        if any(term in v for term in ("working_memory", "n_back", "n-back", "dms")):
            canonical.add("working_memory")
        if any(term in v for term in ("reward", "reinforcement", "q_learning")):
            canonical.add("reward_learning")
        if any(term in v for term in ("visual", "grating", "orientation", "object")):
            canonical.add("visual_stimulation")
        if "sleep" in v:
            canonical.add("sleep")
        if any(term in v for term in ("reach", "motor")):
            canonical.add("reaching")
        if any(term in v for term in ("resting", "rest")):
            canonical.add("resting_state")
        if "fear" in v:
            canonical.add("fear_conditioning")
    return canonical or values


def _dimension_match(expected: set[str], observed: set[str]) -> bool | None:
    if not expected:
        return None
    if "any" in expected:
        return True
    if not observed:
        return False
    return bool(expected & observed)


def _raw_data_required(packet: EvidencePacket) -> bool:
    text = " ".join(
        [
            packet.query_text,
            " ".join(packet.expected_analysis_affordances),
            " ".join(packet.hard_negatives),
        ]
    ).lower()
    return any(
        term in text
        for term in (
            "raw ap",
            "ap-band",
            "ap band",
            "raw_ap_band",
            "raw electrophysiology",
            "spike sorting",
            "kilosort",
        )
    )


def _dimension_evidence(packet: EvidencePacket) -> dict[str, tuple[bool | None, str]]:
    expected_species = _canonical_species(
        _normalise_values(packet.expected_species) | _infer_expected_from_text(packet.query_text, "species")
    )
    observed_species = _canonical_species(_normalise_values(packet.dataset_species))
    expected_modalities = _canonical_modality(
        _normalise_values(packet.expected_modalities) | _infer_expected_from_text(packet.query_text, "modality")
    )
    observed_modalities = _canonical_modality(_normalise_values(packet.dataset_modalities))
    expected_regions = _canonical_regions(
        _normalise_values(packet.expected_brain_regions) | _infer_expected_from_text(packet.query_text, "brain_region")
    )
    observed_regions = _canonical_regions(_normalise_values(packet.dataset_brain_regions))
    expected_tasks = _canonical_tasks(
        _normalise_values(packet.expected_tasks) | _infer_expected_from_text(packet.query_text, "task")
    )
    observed_tasks = _canonical_tasks(_normalise_values(packet.dataset_tasks))
    expected_affordances = (
        _normalise_values(packet.expected_analysis_affordances)
        | _infer_expected_from_text(packet.query_text, "affordance")
    )
    observed_affordances = {
        _normalise_token(match.affordance)
        for match in packet.affordance_matches
        if match.matched
    } | _infer_expected_from_text(f"{packet.title} {packet.description}", "affordance")

    result: dict[str, tuple[bool | None, str]] = {
        "species": (
            _dimension_match(expected_species, observed_species),
            "required" if expected_species else "not_required",
        ),
        "modality": (
            _dimension_match(expected_modalities, observed_modalities),
            "required" if expected_modalities else "not_required",
        ),
        "brain_region": (
            _dimension_match(expected_regions, observed_regions),
            "required" if expected_regions else "not_required",
        ),
        "task": (
            _dimension_match(expected_tasks, observed_tasks),
            "required" if expected_tasks else "not_required",
        ),
        "affordance": (
            _dimension_match(expected_affordances, observed_affordances),
            "required" if expected_affordances else "not_required",
        ),
    }
    if _raw_data_required(packet):
        result["raw_data"] = (packet.has_raw_data is True, "required")
    else:
        result["raw_data"] = (None, "not_required")
    return result


def _completeness_from_dimensions(
    dimension_evidence: dict[str, tuple[bool | None, str]],
) -> tuple[float, list[str], list[str]]:
    required = [
        dimension
        for dimension, (_matched, status) in dimension_evidence.items()
        if status == "required"
    ]
    present = [
        dimension
        for dimension, (matched, status) in dimension_evidence.items()
        if status == "required" and matched is True
    ]
    missing = [
        dimension
        for dimension, (matched, status) in dimension_evidence.items()
        if status == "required" and matched is not True
    ]
    completeness = len(present) / len(required) if required else 0.0
    return round(completeness, 4), present, missing


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from a model response.

    Tolerates two common LLM quirks that broke strict ``json.loads``:
      * leading prose before the object ("Here is my judgment: {...}")
      * trailing commentary after the object ("{...}\\n\\nThis is a good match.")
        which surfaced as ``json.JSONDecodeError: Extra data``.

    Markdown code fences are stripped first. Decoding starts at the first ``{``
    and uses ``raw_decode`` so anything after the matching ``}`` is ignored.
    Raises JudgeParseError if no object can be decoded.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    start = stripped.find("{")
    if start == -1:
        raise JudgeParseError(f"JSON decode error: no object found in response: {stripped[:80]!r}")
    try:
        data, _end = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"JSON decode error: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeParseError(f"JSON decode error: expected object, got {type(data).__name__}")
    return data


def _parse_judgment(
    text: str,
    packet: EvidencePacket,
    model_id: str,
    prompt_version: str,
) -> NeuroJudgment:
    """Parse strict JSON from a model response into NeuroJudgment.

    Raises JudgeParseError on any failure so callers can build an error judgment.
    """
    data: dict[str, Any] = _extract_json_object(text)

    label_raw = data.get("label")
    if label_raw is None:
        raise JudgeParseError("Response missing required field 'label'")
    try:
        label = int(label_raw)
    except (TypeError, ValueError) as exc:
        raise JudgeParseError(f"'label' not coercible to int: {label_raw!r}") from exc
    if label not in (0, 1, 2, 3):
        raise JudgeParseError(f"'label' out of range 0–3: {label}")

    confidence_raw = data.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise JudgeParseError(f"'confidence' not coercible to float: {confidence_raw!r}") from exc
    if not (0.0 <= confidence <= 1.0):
        raise JudgeParseError(f"'confidence' out of range 0.0–1.0: {confidence}")

    label_provenance = "neuro_judge_rag" if packet.concept_explanation_summary else "neuro_judge"
    dimension_evidence = _dimension_evidence(packet)
    fallback_completeness, fallback_present, fallback_missing = _completeness_from_dimensions(
        dimension_evidence
    )
    raw_completeness = data.get("evidence_completeness", fallback_completeness)
    try:
        evidence_completeness = float(raw_completeness)
    except (TypeError, ValueError) as exc:
        raise JudgeParseError(
            f"'evidence_completeness' not coercible to float: {raw_completeness!r}"
        ) from exc
    if not (0.0 <= evidence_completeness <= 1.0):
        raise JudgeParseError(
            f"'evidence_completeness' out of range 0.0–1.0: {evidence_completeness}"
        )
    present = _as_string_list(data.get("required_dimensions_present")) or fallback_present
    missing = _as_string_list(data.get("required_dimensions_missing")) or fallback_missing
    # Tightened abstain: also trigger when label=3 has no explicit dimension support.
    _fallback_abstain = (label >= 2 and bool(missing)) or (label == 3 and not present)
    abstain_recommended = bool(data.get("abstain_recommended", _fallback_abstain))
    abstain_reason_raw = data.get("abstain_reason")
    if abstain_reason_raw is not None:
        abstain_reason = str(abstain_reason_raw)
    elif abstain_recommended and label == 3 and not present:
        abstain_reason = "label_3_with_no_explicit_dimensions"
    elif abstain_recommended:
        abstain_reason = "label>=2_with_missing_required_dimensions"
    else:
        abstain_reason = None

    return NeuroJudgment(
        query_id=packet.query_id,
        dataset_id=packet.dataset_id,
        label=label,
        confidence=confidence,
        rationale_short=str(data.get("rationale_short") or ""),
        evidence_for=[str(e) for e in (data.get("evidence_for") or [])],
        evidence_against=[str(e) for e in (data.get("evidence_against") or [])],
        missing_information=[str(m) for m in (data.get("missing_information") or [])],
        matched_dimensions=[str(d) for d in (data.get("matched_dimensions") or [])],
        failure_modes=[str(f) for f in (data.get("failure_modes") or [])],
        hard_negative_detected=bool(data.get("hard_negative_detected", False)),
        evidence_completeness=evidence_completeness,
        required_dimensions_present=present,
        required_dimensions_missing=missing,
        abstain_recommended=abstain_recommended,
        abstain_reason=abstain_reason,
        judge_model=model_id,
        prompt_version=prompt_version,
        evidence_packet_hash=packet.packet_hash(),
        label_provenance=label_provenance,
    )


def _error_judgment(
    packet: EvidencePacket,
    reason: str,
    model_id: str,
    prompt_version: str,
) -> NeuroJudgment:
    """Return a fallback judgment with label=None replaced by abstain-equivalent."""
    # We cannot store None as label (0–3 required), so we use confidence=0 + rationale.
    return NeuroJudgment(
        query_id=packet.query_id,
        dataset_id=packet.dataset_id,
        label=0,
        confidence=0.0,
        rationale_short=f"judge_error: {reason}",
        failure_modes=[f"judge_error: {reason}"],
        judge_model=model_id,
        prompt_version=prompt_version,
        evidence_packet_hash=packet.packet_hash(),
        evidence_completeness=0.0,
        required_dimensions_missing=list(_DIMENSIONS),
        abstain_recommended=True,
        abstain_reason=f"judge_error: {reason}",
        label_provenance="neuro_judge",
    )


# ---------------------------------------------------------------------------
# Mock judge (tests only)
# ---------------------------------------------------------------------------


class MockNeuroJudge:
    """Deterministic mock — no network access, stable output for testing."""

    _model_id = "mock-neuro-judge"
    _prompt_version = PROMPT_VERSION_DEFAULT

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def judge(self, packet: EvidencePacket) -> NeuroJudgment:
        dimension_evidence = _dimension_evidence(packet)
        evidence_completeness, present, missing = _completeness_from_dimensions(
            dimension_evidence
        )
        matched_dims = present.copy()
        failure_modes: list[str] = []
        missing_information: list[str] = [f"missing_required_dimension:{d}" for d in missing]
        evidence_for: list[str] = [f"matched:{d}" for d in present]
        evidence_against: list[str] = []
        hn_detected = False

        species_match = dimension_evidence["species"][0]
        modality_match = dimension_evidence["modality"][0]
        region_match = dimension_evidence["brain_region"][0]
        task_match = dimension_evidence["task"][0]
        affordance_match = dimension_evidence["affordance"][0]
        raw_match = dimension_evidence["raw_data"][0]
        raw_required = dimension_evidence["raw_data"][1] == "required"

        query_modalities = _canonical_modality(
            _normalise_values(packet.expected_modalities)
            | _infer_expected_from_text(packet.query_text, "modality")
        )
        dataset_modalities = _canonical_modality(_normalise_values(packet.dataset_modalities))
        query_species = _canonical_species(
            _normalise_values(packet.expected_species)
            | _infer_expected_from_text(packet.query_text, "species")
        )
        dataset_species = _canonical_species(_normalise_values(packet.dataset_species))

        if packet.known_failure_warnings or packet.concept_hard_negative_conflicts:
            label, confidence = 0, 0.85
            rationale = "mock: hard-negative signal detected"
            hn_detected = True
            failure_modes.append("mock_rule:hard_negative")
            evidence_against.extend(packet.known_failure_warnings[:2])
            evidence_against.extend(packet.concept_hard_negative_conflicts[:2])
        elif modality_match is False:
            label = 0
            confidence = 0.82
            rationale = "mock: wrong modality for the requested analysis"
            failure_modes.append("mock_rule:wrong_modality")
            evidence_against.append("modality mismatch")
        elif (
            "extracellular_ephys" in query_modalities
            and "calcium_imaging" in dataset_modalities
        ):
            label = 0
            confidence = 0.82
            rationale = "mock: calcium events do not satisfy extracellular spike queries"
            failure_modes.append("mock_rule:calcium_not_extracellular_spikes")
            evidence_against.append("calcium imaging is not extracellular electrophysiology")
        elif (
            ("mouse" in query_species or "rat" in query_species)
            and "human" in dataset_species
            and "extracellular_ephys" in query_modalities
        ):
            label = 0
            confidence = 0.84
            rationale = "mock: human imaging is not interchangeable with rodent electrophysiology"
            failure_modes.append("mock_rule:human_imaging_not_rodent_ephys")
            evidence_against.append("species/modality mismatch")
        elif species_match is False and modality_match is False:
            label = 0
            confidence = 0.8
            rationale = "mock: wrong species and modality"
            failure_modes.append("mock_rule:species_and_modality_mismatch")
        elif packet.query_intent == "REPLICATION" and region_match is False:
            label = 1
            confidence = 0.72
            rationale = "mock: replication query has correct broad signals but wrong target region"
            failure_modes.append("mock_rule:wrong_region_for_replication")
            evidence_against.append("brain region mismatch")
        elif species_match is False:
            label = 1
            confidence = 0.68
            rationale = "mock: species mismatch prevents direct relevance"
            failure_modes.append("mock_rule:species_mismatch")
        elif task_match is False:
            label = 1 if packet.query_intent == "REPLICATION" else 2
            confidence = 0.66
            rationale = "mock: task evidence does not match the query"
            failure_modes.append("mock_rule:task_mismatch")
            evidence_against.append("task mismatch")
        elif raw_required and raw_match is not True:
            label = 2
            confidence = 0.62
            rationale = "mock: correct core match but explicit raw data evidence is missing"
            failure_modes.append("mock_rule:missing_raw_data")
            if "raw_data" not in missing:
                missing.append("raw_data")
                missing_information.append("missing_required_dimension:raw_data")
            evidence_against.append("raw data availability not explicit")
        elif affordance_match is False:
            label = 2
            confidence = 0.64
            rationale = "mock: required analysis affordance is not evidenced"
            failure_modes.append("mock_rule:missing_affordance")
            evidence_against.append("analysis affordance missing")
        elif missing:
            label = 2
            confidence = min(0.7, 0.52 + 0.08 * len(present))
            rationale = "mock: partial match with missing required evidence"
            failure_modes.append("mock_rule:missing_required_evidence")
        else:
            label = 3
            confidence = 0.9
            rationale = "mock: direct match on required dimensions"
            failure_modes.append("mock_rule:direct_match")

        if dimension_evidence["modality"][1] == "required" and modality_match is False:
            confidence = min(confidence, 0.82)
        if dimension_evidence["modality"][1] == "required" and not packet.dataset_modalities:
            confidence = min(confidence, 0.55)
            failure_modes.append("mock_rule:missing_modality_evidence")
        if dimension_evidence["species"][1] == "required" and not packet.dataset_species:
            confidence = min(confidence, 0.6)
            failure_modes.append("mock_rule:missing_species_evidence")
        if raw_required and raw_match is not True:
            confidence = min(confidence, 0.65)

        abstain_recommended = (label >= 2 and bool(missing)) or (label == 3 and not present)
        if label == 3 and not present:
            abstain_reason: str | None = "label_3_with_no_explicit_dimensions"
        elif abstain_recommended:
            abstain_reason = "label>=2_with_missing_required_dimensions"
        else:
            abstain_reason = None

        label_provenance = (
            "neuro_judge_rag" if packet.concept_explanation_summary else "neuro_judge"
        )
        return NeuroJudgment(
            query_id=packet.query_id,
            dataset_id=packet.dataset_id,
            label=label,
            confidence=confidence,
            rationale_short=rationale,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            missing_information=missing_information,
            matched_dimensions=matched_dims,
            failure_modes=failure_modes,
            hard_negative_detected=hn_detected,
            evidence_completeness=evidence_completeness,
            required_dimensions_present=present,
            required_dimensions_missing=missing,
            abstain_recommended=abstain_recommended,
            abstain_reason=abstain_reason,
            judge_model=self._model_id,
            prompt_version=self._prompt_version,
            evidence_packet_hash=packet.packet_hash(),
            label_provenance=label_provenance,
        )


# ---------------------------------------------------------------------------
# Anthropic judge
# ---------------------------------------------------------------------------


class AnthropicNeuroJudge:
    """Judge using the Anthropic Messages API. Skips gracefully without API key."""

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("NEURO_JUDGE_MODEL", self.DEFAULT_MODEL)
        self._client: Any = None
        self._available = False
        self._init()

    def _init(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return
        try:
            import anthropic  # type: ignore[import-not-found]  # noqa: T201
            self._client = anthropic.Anthropic(api_key=api_key)
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    def judge(self, packet: EvidencePacket) -> NeuroJudgment:
        if not self._available:
            return _error_judgment(
                packet,
                "anthropic judge unavailable (no ANTHROPIC_API_KEY or anthropic pkg)",
                self._model,
                PROMPT_VERSION,
            )
        prompt = build_judge_prompt(packet, self.prompt_version)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=768,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return _parse_judgment(text, packet, self._model, PROMPT_VERSION)
        except JudgeParseError as exc:
            return _error_judgment(packet, str(exc), self._model, PROMPT_VERSION)
        except Exception as exc:  # noqa: BLE001
            return _error_judgment(packet, f"api_error: {exc}", self._model, PROMPT_VERSION)


# ---------------------------------------------------------------------------
# OpenAI-compatible judge
# ---------------------------------------------------------------------------


class OpenAINeuroJudge:
    """Judge using any OpenAI-compatible API (OpenAI, Together, Ollama, etc.)."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or os.environ.get("NEURO_JUDGE_MODEL", self.DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client: Any = None
        self._available = False
        self._init()

    def _init(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return
        try:
            import openai  # type: ignore[import-not-found]  # noqa: T201
            kwargs: dict[str, Any] = {"api_key": api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    def judge(self, packet: EvidencePacket) -> NeuroJudgment:
        if not self._available:
            return _error_judgment(
                packet,
                "openai judge unavailable (no OPENAI_API_KEY or openai pkg)",
                self._model,
                PROMPT_VERSION,
            )
        prompt = build_judge_prompt(packet, self.prompt_version)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=768,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
            return _parse_judgment(text, packet, self._model, PROMPT_VERSION)
        except JudgeParseError as exc:
            return _error_judgment(packet, str(exc), self._model, PROMPT_VERSION)
        except Exception as exc:  # noqa: BLE001
            return _error_judgment(packet, f"api_error: {exc}", self._model, PROMPT_VERSION)


# ---------------------------------------------------------------------------
# Gemini judge
# ---------------------------------------------------------------------------


class GeminiNeuroJudge:
    """Judge using Google Gemini generateContent REST API.

    Uses the stdlib HTTP client so the validation path does not require another
    SDK dependency. The API key is read from GEMINI_API_KEY, or GOOGLE_API_KEY
    as a fallback.
    """

    DEFAULT_MODEL = "gemini-3.5-flash"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._model = model or os.environ.get("NEURO_JUDGE_MODEL", self.DEFAULT_MODEL)
        self._base_url = (
            base_url or os.environ.get("GEMINI_API_BASE") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._available = bool(self._api_key)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    def _endpoint(self) -> str:
        return f"{self._base_url}/models/{self._model}:generateContent"

    def _request(self, prompt: str) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 768,
                "temperature": 0,
                "responseFormat": {"text": {"mimeType": "application/json"}},
            },
        }
        encoded = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(),
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key or "",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"gemini_http_{exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"gemini_connection_error: {exc.reason}") from exc

        data = json.loads(response_body)
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("gemini_empty_candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [str(part.get("text", "")) for part in parts if part.get("text")]
        if not texts:
            raise RuntimeError("gemini_empty_text")
        return "\n".join(texts)

    def judge(self, packet: EvidencePacket) -> NeuroJudgment:
        if not self._available:
            return _error_judgment(
                packet,
                "gemini judge unavailable (no GEMINI_API_KEY or GOOGLE_API_KEY)",
                self._model,
                PROMPT_VERSION,
            )
        prompt = build_judge_prompt(packet, self.prompt_version)
        try:
            text = self._request(prompt)
            return _parse_judgment(text, packet, self._model, PROMPT_VERSION)
        except JudgeParseError as exc:
            return _error_judgment(packet, str(exc), self._model, PROMPT_VERSION)
        except Exception as exc:  # noqa: BLE001
            return _error_judgment(packet, f"api_error: {exc}", self._model, PROMPT_VERSION)


# ---------------------------------------------------------------------------
# Local HuggingFace judge
# ---------------------------------------------------------------------------


class LocalHFNeuroJudge:
    """Judge using a local HuggingFace causal language model.

    Requires ``transformers`` and ``torch``. Skips gracefully if unavailable.
    Supports 4-bit and 8-bit quantization via bitsandbytes when requested.
    """

    def __init__(
        self,
        model_name_or_path: str,
        quantization: str | None = None,  # "4bit", "8bit", or None
        max_new_tokens: int = 768,
        device: str = "auto",
    ) -> None:
        self._model_name = model_name_or_path
        self._quantization = quantization
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._pipeline: Any = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline  # noqa: T201

            kwargs: dict[str, Any] = {
                "task": "text-generation",
                "model": self._model_name,
                "device_map": self._device,
                "return_full_text": False,
            }
            if self._quantization == "4bit":
                from transformers import BitsAndBytesConfig  # noqa: T201

                kwargs["model_kwargs"] = {
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True)
                }
            elif self._quantization == "8bit":
                from transformers import BitsAndBytesConfig  # noqa: T201

                kwargs["model_kwargs"] = {
                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
                }
            self._pipeline = hf_pipeline(**kwargs)
            self._available = True
        except (ImportError, Exception):  # noqa: BLE001
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    def judge(self, packet: EvidencePacket) -> NeuroJudgment:
        if not self._available:
            return _error_judgment(
                packet,
                "local_hf judge unavailable (transformers/torch not installed or model load failed)",
                self._model_name,
                PROMPT_VERSION,
            )
        prompt = build_judge_prompt(packet, self.prompt_version)
        try:
            outputs = self._pipeline(prompt, max_new_tokens=self._max_new_tokens)
            text = outputs[0]["generated_text"]
            return _parse_judgment(text, packet, self._model_name, PROMPT_VERSION)
        except JudgeParseError as exc:
            return _error_judgment(packet, str(exc), self._model_name, PROMPT_VERSION)
        except Exception as exc:  # noqa: BLE001
            return _error_judgment(
                packet, f"generation_error: {exc}", self._model_name, PROMPT_VERSION
            )


# ---------------------------------------------------------------------------
# BrainGPT adapter
# ---------------------------------------------------------------------------


class BrainGPTAdapter(LocalHFNeuroJudge):
    """BrainGPT adapter over a Mistral-7B checkpoint.

    Skips gracefully if the BrainGPT model weights are not installed.
    Tests must not require BrainGPT; use MockNeuroJudge instead.
    """

    BRAINGPT_MODEL_ID = "braingpt/BrainGPT-Mistral-7B"

    def __init__(
        self,
        model_name_or_path: str | None = None,
        quantization: str | None = "4bit",
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path or self.BRAINGPT_MODEL_ID,
            quantization=quantization,
        )

    @property
    def model_id(self) -> str:
        return f"braingpt:{self._model_name}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKENDS = frozenset({"mock", "anthropic", "openai", "gemini", "local_hf", "braingpt"})


def build_neuro_judge(
    backend: str = "anthropic",
    **kwargs: Any,
) -> NeuroJudgeProtocol:
    """Return a NeuroJudgeProtocol implementation for the requested backend.

    Args:
        backend: One of "mock", "anthropic", "openai", "gemini", "local_hf", "braingpt".
        **kwargs: Forwarded to the judge constructor.
    """
    backend = backend.lower()
    if backend == "mock":
        return MockNeuroJudge()
    if backend == "anthropic":
        return AnthropicNeuroJudge(**kwargs)
    if backend == "openai":
        return OpenAINeuroJudge(**kwargs)
    if backend == "gemini":
        return GeminiNeuroJudge(**kwargs)
    if backend == "local_hf":
        return LocalHFNeuroJudge(**kwargs)
    if backend == "braingpt":
        return BrainGPTAdapter(**kwargs)
    raise ValueError(f"Unknown backend {backend!r}. Choose from: {sorted(_BACKENDS)}")
