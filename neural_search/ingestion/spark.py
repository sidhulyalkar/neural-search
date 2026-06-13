"""SPARK (SFARI Pilot Awards for Research on Autism) ingestion adapter.

SPARK is a large-scale autism research initiative managed by the Simons
Foundation Autism Research Initiative (SFARI). Datasets are available via
SFARI Base, which requires credentialed access.

This adapter provides:
- A local normalization function for raw SPARK-format records.
- A stub fetcher that returns 20 representative synthetic records for
  development and testing. Live SFARI Base API access is not attempted here
  because it requires institutional credentials.

Live API reference: https://base.sfari.org  (auth required)
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.registry import register
from neural_search.normalized import (
    evidence_label_from_extraction,
    stable_normalized_id,
)
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord, UsabilityFlags

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic stub records (no network calls)
# ---------------------------------------------------------------------------

_SPARK_STUB_RECORDS: list[dict[str, Any]] = [
    {
        "source_id": "SPARK-FMRI-001",
        "title": "SPARK fMRI Social Brain Circuits in Autism Spectrum Disorder",
        "description": (
            "Resting-state and task fMRI data from 312 autistic participants and "
            "148 neurotypical controls. Tasks include social cognition, theory of mind, "
            "and biological motion perception. BIDS-formatted with derivatives."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-FMRI-001",
        "modalities": ["fmri"],
        "brain_regions": ["prefrontal_cortex", "anterior_cingulate", "amygdala"],
        "tasks": ["social_cognition", "theory_of_mind"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 460,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-EEG-001",
        "title": "SPARK EEG Sensory Processing and Attention in ASD",
        "description": (
            "High-density EEG (128 ch) from 200 autistic children and 100 typically "
            "developing controls during visual and auditory attention tasks. "
            "Event-related potentials and oscillatory measures available."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-EEG-001",
        "modalities": ["eeg"],
        "brain_regions": ["prefrontal_cortex", "anterior_cingulate"],
        "tasks": ["attention", "sensory_processing"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 300,
        "has_processed_data": True,
        "has_raw_data": True,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-BEHAV-001",
        "title": "SPARK Behavioral Battery: Executive Function in Autism",
        "description": (
            "Comprehensive behavioral assessments in 1,850 autistic participants and "
            "their first-degree relatives. Measures include inhibitory control, "
            "cognitive flexibility, working memory, and set-shifting tasks. NDA format."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-BEHAV-001",
        "modalities": ["behavior_video"],
        "brain_regions": ["prefrontal_cortex", "striatum"],
        "tasks": ["executive_function", "working_memory"],
        "data_standards": ["NDA"],
        "n_subjects": 1850,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "SFARI Open Access",
    },
    {
        "source_id": "SPARK-EYE-001",
        "title": "SPARK Eye-Tracking Social Attention Cohort",
        "description": (
            "Eye-tracking data from 520 autistic and 260 neurotypical participants "
            "viewing social scenes and faces. Gaze fixation, saccade metrics, and "
            "pupillometry during social and non-social video stimuli."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-EYE-001",
        "modalities": ["behavior_video"],
        "brain_regions": ["amygdala", "prefrontal_cortex"],
        "tasks": ["social_cognition", "attention"],
        "data_standards": ["NDA"],
        "n_subjects": 780,
        "has_processed_data": True,
        "has_raw_data": True,
        "license": "SFARI Open Access",
    },
    {
        "source_id": "SPARK-GENO-001",
        "title": "SPARK Whole-Exome Sequencing: De Novo Variants in ASD",
        "description": (
            "Whole-exome sequencing from 10,000 autistic probands and their "
            "biological parents. Variant annotation, CNV calls, and polygenic scores. "
            "Includes phenotypic metadata from SPARK registration questionnaires."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-GENO-001",
        "modalities": [],
        "brain_regions": [],
        "tasks": [],
        "data_standards": ["NDA"],
        "n_subjects": 10000,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "SFARI Research Access",
        "additional_metadata": {"data_type": "genetics", "sequencing_type": "WES"},
    },
    {
        "source_id": "SPARK-FMRI-002",
        "title": "SPARK fMRI Executive Function and Cognitive Control in ASD",
        "description": (
            "Task-based fMRI during go/no-go, stop-signal, and n-back paradigms in "
            "180 autistic adults and 90 matched controls. Preprocessed derivatives "
            "in MNI space; raw DICOM available via credentialed request."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-FMRI-002",
        "modalities": ["fmri"],
        "brain_regions": ["prefrontal_cortex", "anterior_cingulate", "striatum"],
        "tasks": ["executive_function", "inhibitory_control", "working_memory"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 270,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-EEG-002",
        "title": "SPARK Resting EEG and Event-Related Potentials: Social Prediction",
        "description": (
            "64-channel resting-state EEG and social prediction ERP paradigm from "
            "150 autistic children (ages 6–12) and 80 age-matched controls. "
            "Data in BIDS EEG format with PREP-pipeline preprocessing derivatives."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-EEG-002",
        "modalities": ["eeg"],
        "brain_regions": ["prefrontal_cortex", "anterior_cingulate"],
        "tasks": ["social_cognition", "prediction"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 230,
        "has_processed_data": True,
        "has_raw_data": True,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-BEHAV-002",
        "title": "SPARK Longitudinal Adaptive Behavior Assessments",
        "description": (
            "Longitudinal behavioral assessments (Vineland-3, ABAS-3) from 2,400 "
            "autistic participants across three timepoints. Includes caregiver-reported "
            "adaptive behavior, restricted and repetitive behaviors, and sensory sensitivity."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-BEHAV-002",
        "modalities": ["behavior_video"],
        "brain_regions": [],
        "tasks": ["adaptive_behavior"],
        "data_standards": ["NDA"],
        "n_subjects": 2400,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "SFARI Open Access",
    },
    {
        "source_id": "SPARK-EYE-002",
        "title": "SPARK Infant Sibling Eye-Tracking: Early Social Attention",
        "description": (
            "Longitudinal eye-tracking from 340 infant siblings of autistic probands "
            "at 6, 12, and 24 months. Social orienting, joint attention, and face "
            "preference paradigms. Prospective ASD outcome data included."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-EYE-002",
        "modalities": ["behavior_video"],
        "brain_regions": [],
        "tasks": ["social_cognition", "attention"],
        "data_standards": ["NDA"],
        "n_subjects": 340,
        "has_processed_data": True,
        "has_raw_data": True,
        "license": "SFARI Research Access",
    },
    {
        "source_id": "SPARK-FMRI-003",
        "title": "SPARK fMRI Amygdala Reactivity to Social and Non-Social Threats",
        "description": (
            "Task fMRI examining amygdala and prefrontal reactivity to social threat "
            "stimuli in 95 autistic and 60 neurotypical adults. Threat conditioning, "
            "extinction, and recall blocks. BIDS-formatted with FSL derivatives."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-FMRI-003",
        "modalities": ["fmri"],
        "brain_regions": ["amygdala", "prefrontal_cortex"],
        "tasks": ["social_cognition", "threat_conditioning"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 155,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-MULTI-001",
        "title": "SPARK Multimodal ASD Characterization: fMRI, EEG, and Behavior",
        "description": (
            "Concurrent fMRI, EEG, and behavioral assessments in 80 autistic and "
            "40 neurotypical adults. Resting state and social cognition tasks. "
            "Synchronized physiological data and eye-tracking co-registration."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-MULTI-001",
        "modalities": ["fmri", "eeg", "behavior_video"],
        "brain_regions": ["prefrontal_cortex", "anterior_cingulate", "amygdala"],
        "tasks": ["social_cognition", "attention"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 120,
        "has_processed_data": True,
        "has_raw_data": True,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-BEHAV-003",
        "title": "SPARK Social Communication Questionnaire Longitudinal Database",
        "description": (
            "Standardized caregiver-reported social communication assessments (SCQ, "
            "SRS-2, VABS) from 5,200 SPARK participants over 5 years. "
            "Age range 2–18 years; linked to genetic and clinical phenotype data."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-BEHAV-003",
        "modalities": ["behavior_video"],
        "brain_regions": [],
        "tasks": ["social_cognition"],
        "data_standards": ["NDA"],
        "n_subjects": 5200,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "SFARI Open Access",
    },
    {
        "source_id": "SPARK-EEG-003",
        "title": "SPARK EEG Gamma-Band Oscillations and Sensory Hyper-Reactivity",
        "description": (
            "Auditory steady-state response and gamma-band EEG in 110 autistic "
            "and 55 neurotypical adolescents. Passive listening and active oddball "
            "paradigms; 64-channel EEG plus auditory brainstem responses."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-EEG-003",
        "modalities": ["eeg"],
        "brain_regions": ["anterior_cingulate"],
        "tasks": ["attention", "sensory_processing"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 165,
        "has_processed_data": True,
        "has_raw_data": True,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-EYE-003",
        "title": "SPARK Eye-Tracking Gaze Contingency and Joint Attention",
        "description": (
            "Gaze-contingent eye-tracking paradigm from 200 autistic children "
            "and 120 neurotypical peers. Joint attention bids, declarative pointing "
            "response, and social referencing with real-time gaze feedback."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-EYE-003",
        "modalities": ["behavior_video"],
        "brain_regions": [],
        "tasks": ["social_cognition", "joint_attention"],
        "data_standards": ["NDA"],
        "n_subjects": 320,
        "has_processed_data": True,
        "has_raw_data": True,
        "license": "SFARI Open Access",
    },
    {
        "source_id": "SPARK-FMRI-004",
        "title": "SPARK fMRI Default Mode Network Connectivity in ASD Subtypes",
        "description": (
            "Resting-state fMRI in 420 autistic participants stratified by cognitive "
            "profile and 210 controls. Independent component analysis of default mode, "
            "salience, and fronto-parietal networks. ABIDE-compatible preprocessing."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-FMRI-004",
        "modalities": ["fmri"],
        "brain_regions": ["prefrontal_cortex", "anterior_cingulate"],
        "tasks": ["resting_state"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 630,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-GENO-002",
        "title": "SPARK Genome-Wide Association Study: ASD Polygenic Architecture",
        "description": (
            "GWAS data from 15,000 autistic individuals and 22,000 controls. "
            "Imputed genotypes, principal components, and polygenic risk scores. "
            "Includes IQ, adaptive behavior, and co-occurring condition phenotypes."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-GENO-002",
        "modalities": [],
        "brain_regions": [],
        "tasks": [],
        "data_standards": ["NDA"],
        "n_subjects": 37000,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "SFARI Research Access",
        "additional_metadata": {"data_type": "genetics", "sequencing_type": "GWAS"},
    },
    {
        "source_id": "SPARK-BEHAV-004",
        "title": "SPARK Caregiver Stress and Family Functioning in ASD",
        "description": (
            "Parent-reported measures of caregiver stress, family functioning, and "
            "quality of life from 3,100 SPARK families. Paired with child behavioral "
            "severity and autism diagnostic composite scores."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-BEHAV-004",
        "modalities": ["behavior_video"],
        "brain_regions": [],
        "tasks": ["adaptive_behavior"],
        "data_standards": ["NDA"],
        "n_subjects": 3100,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "SFARI Open Access",
    },
    {
        "source_id": "SPARK-EEG-004",
        "title": "SPARK Mu Rhythm Suppression and Mirror Neuron Function in ASD",
        "description": (
            "EEG mu rhythm (8–13 Hz) suppression during action observation and "
            "execution in 75 autistic and 45 neurotypical adults. "
            "32-channel EEG with kinematic motion capture co-registration."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-EEG-004",
        "modalities": ["eeg", "behavior_video"],
        "brain_regions": ["anterior_cingulate"],
        "tasks": ["social_cognition"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 120,
        "has_processed_data": True,
        "has_raw_data": True,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-MULTI-002",
        "title": "SPARK Longitudinal fMRI and Behavioral Trajectories in ASD (Ages 8–16)",
        "description": (
            "Annual fMRI and comprehensive behavioral battery in 130 autistic youth "
            "and 70 controls over 3 years. Social cognition, executive function, "
            "and attention paradigms. Linked to SPARK genetic and phenotypic data."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-MULTI-002",
        "modalities": ["fmri", "behavior_video"],
        "brain_regions": ["prefrontal_cortex", "amygdala", "striatum"],
        "tasks": ["social_cognition", "executive_function"],
        "data_standards": ["BIDS", "NDA"],
        "n_subjects": 200,
        "has_processed_data": True,
        "has_raw_data": False,
        "license": "CC-BY-4.0",
    },
    {
        "source_id": "SPARK-EYE-004",
        "title": "SPARK Eye-Tracking Biological Motion Perception in ASD",
        "description": (
            "Point-light display and biological motion eye-tracking in 180 autistic "
            "and 90 neurotypical participants. Gaze toward social agents vs. scrambled "
            "motion controls. Paired with Autism Diagnostic Observation Schedule scores."
        ),
        "url": "https://base.sfari.org/datasets/SPARK-EYE-004",
        "modalities": ["behavior_video"],
        "brain_regions": [],
        "tasks": ["social_cognition", "attention"],
        "data_standards": ["NDA"],
        "n_subjects": 270,
        "has_processed_data": True,
        "has_raw_data": True,
        "license": "SFARI Open Access",
    },
]


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------


def normalize_spark_record(raw: dict[str, Any]) -> NormalizedDatasetRecord:
    """Normalize a raw SPARK dataset record into the provenance-aware schema.

    Args:
        raw: A SPARK-format metadata dict.  Must include at minimum
             ``source_id`` and ``title``.

    Returns:
        A fully validated :class:`~neural_search.schemas.NormalizedDatasetRecord`.
    """
    source_id = str(raw["source_id"])
    title = str(raw.get("title") or f"SPARK {source_id}")
    description: str | None = raw.get("description")

    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata={
            **raw,
            "standard": " ".join(raw.get("data_standards", [])),
        },
        linked_paper_abstracts=[],
    )

    # Prefer explicit stub fields; fall back to extraction
    raw_modalities: list[str] = raw.get("modalities") or [
        item.id for item in extraction.modalities
    ]
    raw_brain_regions: list[str] = raw.get("brain_regions") or [
        item.id for item in extraction.brain_regions
    ]
    raw_tasks: list[str] = raw.get("tasks") or [item.id for item in extraction.tasks]
    raw_standards: list[str] = raw.get("data_standards") or [
        item.id for item in extraction.data_standards
    ]

    source_value = " ".join(
        str(p) for p in [title, description, raw_brain_regions, raw_tasks] if p
    )

    has_raw = bool(raw.get("has_raw_data", False))
    has_processed = bool(raw.get("has_processed_data", True))
    text_lower = f"{title} {description or ''}".casefold()

    return NormalizedDatasetRecord(
        dataset_id=stable_normalized_id("dataset", "spark", source_id),
        source="spark",
        source_id=source_id,
        title=title,
        description=description,
        url=raw.get("url"),
        species=[
            evidence_label_from_extraction(
                label, "species", source_field="metadata", source_value=source_value
            )
            for label in extraction.species
        ]
        or [
            # SPARK is exclusively human research; always assert species if extractor misses
            EvidenceLabel(
                id="human",
                label="human",
                label_type="species",
                confidence=1.0,
                evidence_text="SPARK is a human autism research initiative",
                source_field="stub",
                source_value="SPARK human cohort",
            )
        ],
        modalities=[
            evidence_label_from_extraction(
                label, "modality", source_field="metadata", source_value=source_value
            )
            for label in extraction.modalities
            if label.id in raw_modalities or not raw_modalities
        ],
        brain_regions=[
            evidence_label_from_extraction(
                label, "brain_region", source_field="metadata", source_value=source_value
            )
            for label in extraction.brain_regions
        ],
        tasks=[
            evidence_label_from_extraction(
                label, "task", source_field="metadata", source_value=source_value
            )
            for label in extraction.tasks
        ],
        behavioral_events=[
            evidence_label_from_extraction(
                label,
                "behavioral_event",
                source_field="metadata",
                source_value=source_value,
            )
            for label in extraction.behaviors
        ],
        data_standards=[
            evidence_label_from_extraction(
                label,
                "data_standard",
                source_field="metadata",
                source_value=source_value,
            )
            for label in extraction.data_standards
        ],
        usability_flags=UsabilityFlags(
            has_trials=any(
                term in text_lower for term in ["trial", "event", "task", "block"]
            ),
            has_behavior=bool(extraction.behaviors)
            or "behavior" in text_lower
            or "behaviour" in text_lower,
            has_neural_data=any(m in raw_modalities for m in ["fmri", "eeg", "meg"]),
            has_raw_data=has_raw,
            has_processed_data=has_processed,
            has_standard_format=bool(
                {"BIDS", "NDA"} & set(raw_standards)
            ),
        ),
        missing_fields=extraction.missing_fields,
    )


# ---------------------------------------------------------------------------
# Stub fetcher (no network calls)
# ---------------------------------------------------------------------------


@register("spark")
def fetch_spark_records(limit: int = 500) -> list[dict[str, Any]]:
    """Return representative SPARK stub records for corpus building and testing.

    This function deliberately does NOT contact the SFARI Base API because
    live access requires institutional credentials.  The 20 stubs cover the
    main modalities (fMRI, EEG, behavioral, genetics, eye-tracking) and are
    suitable for downstream graph construction and retrieval benchmarks.

    Args:
        limit: Maximum number of records to return (ceiling is 20 stubs).

    Returns:
        A list of raw SPARK-format metadata dicts.
    """
    records = _SPARK_STUB_RECORDS[:limit]
    logger.info("SPARK: returning %d stub records (live API requires credentials)", len(records))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — normalize and print SPARK stub records."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.ingestion.spark",
        description="Normalize SPARK stub records and print as JSON.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Print records without writing")
    args = parser.parse_args(argv)

    raw_records = fetch_spark_records(limit=args.limit)
    normalized = [normalize_spark_record(r) for r in raw_records]

    for record in normalized:
        print(json.dumps(record.model_dump(mode="json", exclude_none=True), sort_keys=True))

    if not args.dry_run:
        print(
            f"\n# {len(normalized)} SPARK records normalized "
            "(live SFARI Base API not contacted — credentials required)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
