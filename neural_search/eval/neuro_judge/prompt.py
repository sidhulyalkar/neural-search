"""Prompt construction for the neuro_judge.

build_judge_prompt(packet) → str
"""

from __future__ import annotations

from neural_search.eval.neuro_judge.evidence_packet import EvidencePacket

# ---------------------------------------------------------------------------
# Rubric and rules (shown verbatim inside every prompt)
# ---------------------------------------------------------------------------

NEURO_RUBRIC = """
## Scoring rubric (0–3 scale)

Score 0 — Not relevant or actively misleading. Dataset lacks core requirements.
  May share only surface keywords.
  Examples: human fMRI for rodent place-cell theta query; calcium imaging for
  raw extracellular spike sorting; wrong species AND wrong modality;
  behavior-only when neural recordings are required.

Score 1 — Weakly related. Some broad scientific concept matches, but the
  dataset is unlikely to support the intended analysis.
  Examples: correct species but wrong brain region; correct modality but wrong
  task; related circuit but not the requested target;
  visual MT when query requires LIP accumulator replication.

Score 2 — Useful with caveats. Dataset likely supports part of the goal, but
  important evidence is missing or imperfect.
  Examples: correct species and modality but raw data access uncertain; correct
  task and modality but imperfect brain region; suitable for exploration but not
  direct replication.

Score 3 — Highly relevant. Dataset directly supports the query.
  It has the requested species, modality, brain region/circuit, task, and
  analysis affordance. For pipeline reuse, required raw or processed data must
  plausibly be available. For replication, the match must be close enough to
  reproduce or strongly test the target claim.
"""

NEURO_RULES = """
## Neuroscience-specific rules (MUST apply these)

- fMRI-only datasets are 0 for queries requiring spikes, LFP, theta oscillations,
  place cells, or single-unit physiology.
- Correct modality but wrong brain region is usually 1 for replication queries.
- Correct species + modality with uncertain raw-data availability is usually 2.
- Human imaging and rodent electrophysiology are NOT interchangeable unless the
  query is explicitly cross-species or conceptual.
- MT/V5 is not LIP; PFC is not LIP; entorhinal cortex is not hippocampus —
  unless the query explicitly allows related navigation circuitry.
- Calcium-imaging extracted events are NOT extracellular spikes.
- ALF/processed spike tables are NOT automatically raw AP-band data.
- NWB/BIDS/ALF format is evidence of accessibility, not of suitability.
- Do NOT infer raw data availability unless explicit evidence is present in the
  description or file-format evidence section.
- A hard-negative match MUST produce label=0 regardless of other signals.
- Absent or unknown metadata should lower confidence, not automatically lower
  the relevance score — unless the absent field is required for the query.
"""

PROMPT_TEMPLATE = """\
You are an expert neuroscience dataset relevance judge.

## Query
ID: {query_id}
Text: {query_text}
Intent: {query_intent}

## Query constraints
Expected species: {expected_species}
Expected modalities: {expected_modalities}
Expected brain regions: {expected_brain_regions}
Expected tasks: {expected_tasks}
Expected analysis affordances: {expected_analysis_affordances}

## Hard-negative patterns (any match → label MUST be 0)
{hard_negatives_block}

## Dataset
ID: {dataset_id}
Title: {title}
Source archive: {source_archive}
URL: {source_url}
Description (first 1200 chars): {description}
Species: {dataset_species}
Modalities: {dataset_modalities}
Brain regions: {dataset_brain_regions}
Tasks: {dataset_tasks}
Data standards / formats: {data_standards}
License: {license}
File format evidence: {file_format_evidence}
Has raw data (inferred): {has_raw_data}
Has processed data (inferred): {has_processed_data}

## Metadata signal quality
The following signals were EXPLICITLY stated in the dataset record (high confidence):
{explicit_signals}

The following signals were INFERRED from free text or heuristics (lower confidence):
{inferred_signals}

## Linked papers
{linked_papers_block}

## Affordance evidence
{affordance_block}

## Concept-memory signals
{concept_summary_block}
Matched concepts: {matched_concepts}
Missing evidence from concept memory: {concept_missing}
Concept hard-negative conflicts: {concept_hn_conflicts}

## Known failure warnings (pre-screened)
{warnings_block}
{neuro_rules}
{neuro_rubric}
## Your task

Work through the following steps before writing your final JSON:

STEP 1 — Hard-negative check: Does this dataset match any hard-negative pattern?
  If yes, label MUST be 0. Write why in failure_modes.

STEP 2 — Dimension-by-dimension check. For each required dimension (species,
  modality, brain_region, task, affordance), answer:
  (a) Is there EXPLICIT evidence in the record? (b) Does it match the query?
  Add matched dimensions to required_dimensions_present; missing to required_dimensions_missing.

STEP 3 — Raw data check: Does the query require raw neural recordings?
  Is raw data availability explicitly confirmed, inferred, or unknown?
  If uncertain for a high-scoring candidate, recommend abstain.

STEP 4 — Evidence completeness: What fraction of required dimensions have
  explicit (not inferred) evidence? This is your evidence_completeness score.

STEP 5 — Final score: Apply the rubric. A dataset with all dimensions matched
  and explicit raw data evidence is 3. One with good matches but uncertain raw
  data or one missing dimension is 2. One with only broad matches is 1.
  Any hard-negative match or completely wrong modality/species is 0.

After working through these steps, return ONLY valid JSON with these fields:

{{
  "reasoning_trace": "<brief dimension-by-dimension reasoning from steps 1-5, 2-4 sentences>",
  "label": <integer 0, 1, 2, or 3>,
  "confidence": <float 0.0–1.0>,
  "rationale_short": "<one or two sentences summarising why this score>",
  "evidence_for": ["<specific evidence strings supporting relevance>"],
  "evidence_against": ["<specific evidence strings against relevance>"],
  "missing_information": ["<fields or data needed to confirm relevance>"],
  "matched_dimensions": ["species", "modality", "brain_region", "task", "affordance" — whichever explicitly match],
  "failure_modes": ["<any failure modes detected, e.g. wrong_species, no_raw_data>"],
  "hard_negative_detected": <true or false>,
  "evidence_completeness": <float 0.0–1.0, fraction of required dimensions with EXPLICIT evidence>,
  "required_dimensions_present": ["<required dimensions with explicit support in the record>"],
  "required_dimensions_missing": ["<required dimensions absent or only inferred>"],
  "abstain_recommended": <true if label >= 2 but critical required evidence is missing or only inferred>,
  "abstain_reason": "<short reason string, or null if abstain_recommended is false>"
}}
"""

PROMPT_VERSION = "v2"


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _fmt_list(items: list[str], default: str = "none") -> str:
    return ", ".join(items) if items else default


def _fmt_block(items: list[str], default: str = "none") -> str:
    if not items:
        return default
    return "\n".join(f"  - {i}" for i in items)


def _build_signal_quality(packet: EvidencePacket) -> tuple[str, str]:
    """Return (explicit_signals, inferred_signals) formatted blocks.

    Explicit: came from structured metadata fields (species, modalities, tasks, etc.)
    Inferred: derived from free text, file format heuristics, or description keywords.
    """
    explicit: list[str] = []
    inferred: list[str] = []

    if packet.dataset_species:
        explicit.append(f"species: {', '.join(packet.dataset_species)}")
    if packet.dataset_modalities:
        explicit.append(f"modalities: {', '.join(packet.dataset_modalities)}")
    if packet.dataset_brain_regions:
        explicit.append(f"brain_regions: {', '.join(packet.dataset_brain_regions)}")
    if packet.dataset_tasks:
        explicit.append(f"tasks: {', '.join(packet.dataset_tasks)}")
    if packet.data_standards:
        explicit.append(f"data_standards: {', '.join(packet.data_standards)}")

    # has_raw_data / has_processed_data come from heuristic keyword matching
    if packet.has_raw_data is not None:
        inferred.append(f"has_raw_data={packet.has_raw_data} (keyword heuristic)")
    if packet.has_processed_data is not None:
        inferred.append(f"has_processed_data={packet.has_processed_data} (keyword heuristic)")
    if packet.file_format_evidence:
        inferred.append(f"file formats found in text: {', '.join(packet.file_format_evidence)}")
    if packet.concept_explanation_summary:
        inferred.append("concept-memory summary available (RAG-derived)")

    return (
        _fmt_block(explicit, default="none explicitly stated"),
        _fmt_block(inferred, default="none inferred"),
    )


def build_judge_prompt(packet: EvidencePacket, prompt_version: str = PROMPT_VERSION) -> str:
    """Render the full judge prompt from an EvidencePacket."""
    linked_papers_block = "none"
    if packet.linked_papers:
        parts = []
        for p in packet.linked_papers:
            parts.append(f"  Title: {p.title}")
            if p.abstract:
                parts.append(f"  Abstract (first 500): {p.abstract[:500]}")
        linked_papers_block = "\n".join(parts)

    affordance_block = "none"
    if packet.affordance_matches:
        parts = []
        for am in packet.affordance_matches:
            status = "MATCHED" if am.matched else "NOT MATCHED"
            parts.append(f"  [{status}] {am.affordance} (confidence={am.confidence:.2f})")
            if am.missing_requirements:
                parts.append(f"    Missing: {', '.join(am.missing_requirements)}")
            if am.rationale:
                parts.append(f"    Rationale: {am.rationale}")
        affordance_block = "\n".join(parts)

    concept_summary_block = (
        f"Summary: {packet.concept_explanation_summary}"
        if packet.concept_explanation_summary
        else "No concept-memory evidence available."
    )

    explicit_signals, inferred_signals = _build_signal_quality(packet)

    return PROMPT_TEMPLATE.format(
        query_id=packet.query_id,
        query_text=packet.query_text,
        query_intent=packet.query_intent or "unspecified",
        expected_species=_fmt_list(packet.expected_species),
        expected_modalities=_fmt_list(packet.expected_modalities),
        expected_brain_regions=_fmt_list(packet.expected_brain_regions),
        expected_tasks=_fmt_list(packet.expected_tasks),
        expected_analysis_affordances=_fmt_list(packet.expected_analysis_affordances),
        hard_negatives_block=_fmt_block(packet.hard_negatives),
        dataset_id=packet.dataset_id,
        title=packet.title,
        source_archive=packet.source_archive,
        source_url=packet.source_url,
        description=(packet.description[:1200] if packet.description else "none"),
        dataset_species=_fmt_list(packet.dataset_species),
        dataset_modalities=_fmt_list(packet.dataset_modalities),
        dataset_brain_regions=_fmt_list(packet.dataset_brain_regions),
        dataset_tasks=_fmt_list(packet.dataset_tasks),
        data_standards=_fmt_list(packet.data_standards),
        license=packet.license or "unknown",
        file_format_evidence=_fmt_list(packet.file_format_evidence),
        has_raw_data=str(packet.has_raw_data),
        has_processed_data=str(packet.has_processed_data),
        explicit_signals=explicit_signals,
        inferred_signals=inferred_signals,
        linked_papers_block=linked_papers_block,
        affordance_block=affordance_block,
        concept_summary_block=concept_summary_block,
        matched_concepts=_fmt_list(packet.matched_concept_names),
        concept_missing=_fmt_block(packet.concept_missing_evidence),
        concept_hn_conflicts=_fmt_block(packet.concept_hard_negative_conflicts),
        warnings_block=_fmt_block(packet.known_failure_warnings),
        neuro_rules=NEURO_RULES,
        neuro_rubric=NEURO_RUBRIC,
    )
