"""Seed data for the v0.1 field-state layer."""

from __future__ import annotations

from pathlib import Path

from neural_search.field_state.schemas import (
    BenchmarkGap,
    ClaimStatus,
    EvidenceLevel,
    FieldClaim,
    FieldOpportunity,
    GapStatus,
)
from neural_search.field_state.scoring import rank_opportunities

KNOWN_INPUTS = [
    "data/corpus/normalized/combined_corpus.jsonl",
    "reports/eval",
    "artifacts/qrels.jsonl",
    "docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md",
    "docs/whitepaper/neural_search_whitepaper.tex",
]


def existing_inputs(root: Path | None = None) -> list[str]:
    """Return known field-state input artifacts that exist locally."""
    base = root or Path.cwd()
    found: list[str] = []
    for item in KNOWN_INPUTS:
        if (base / item).exists():
            found.append(item)
    return found


def seed_claims(root: Path | None = None) -> list[FieldClaim]:
    """Return initial field-level claims to track."""
    artifacts = existing_inputs(root)
    corpus_artifacts = [
        item for item in artifacts if item.startswith("data/corpus") or item == "reports/eval"
    ]
    qrels_artifacts = [item for item in artifacts if "qrels" in item or item == "reports/eval"]
    whitepaper_artifacts = [
        item for item in artifacts if "whitepaper" in item.lower() or "WHITEPAPER" in item
    ]

    return [
        FieldClaim(
            claim_id="claim_dense_semantic_retrieval",
            claim_text="Dense semantic retrieval improves dataset matching.",
            evidence_level=EvidenceLevel.PLAUSIBLE,
            confidence=0.62,
            related_artifacts=corpus_artifacts + whitepaper_artifacts,
            missing_tests=[
                "Compare dense retrieval against lexical and metadata-only baselines on human qrels.",
                "Measure whether gains hold for dataset-method compatibility queries.",
            ],
            status=ClaimStatus.NEEDS_VALIDATION,
        ),
        FieldClaim(
            claim_id="claim_hard_negatives",
            claim_text="Hard negatives are necessary for scientific retrieval evaluation.",
            evidence_level=EvidenceLevel.SUPPORTED,
            confidence=0.72,
            related_artifacts=["reports/eval"] if "reports/eval" in artifacts else [],
            missing_tests=[
                "Track top-k hard-negative violations by query type.",
                "Separate near-miss datasets from clearly irrelevant datasets.",
            ],
            status=ClaimStatus.PARTIALLY_TESTED,
        ),
        FieldClaim(
            claim_id="claim_human_qrels",
            claim_text="Human qrels are required for credible nDCG/MRR metrics.",
            evidence_level=EvidenceLevel.PLAUSIBLE,
            confidence=0.78,
            related_artifacts=qrels_artifacts,
            missing_tests=[
                "Create expert-labeled qrels for dataset-method compatibility.",
                "Report inter-annotator agreement or adjudication notes.",
            ],
            status=ClaimStatus.NEEDS_VALIDATION,
        ),
        FieldClaim(
            claim_id="claim_metadata_richness",
            claim_text="Metadata richness affects dataset reuse potential.",
            evidence_level=EvidenceLevel.PLAUSIBLE,
            confidence=0.66,
            related_artifacts=corpus_artifacts,
            missing_tests=[
                "Define metadata richness features.",
                "Estimate relationship between metadata richness and reuse/readiness labels.",
            ],
            status=ClaimStatus.NEEDS_VALIDATION,
        ),
        FieldClaim(
            claim_id="claim_affordance_extraction",
            claim_text="Analysis affordance extraction improves dataset recommendation.",
            evidence_level=EvidenceLevel.PLAUSIBLE,
            confidence=0.64,
            related_artifacts=corpus_artifacts + whitepaper_artifacts,
            missing_tests=[
                "Build a small gold benchmark for analysis affordances.",
                "Evaluate recommendation quality with and without affordance features.",
            ],
            status=ClaimStatus.NEEDS_VALIDATION,
        ),
        FieldClaim(
            claim_id="claim_graph_proximity",
            claim_text="Graph proximity may improve dataset-method matching.",
            evidence_level=EvidenceLevel.HYPOTHESIS,
            confidence=0.55,
            related_artifacts=whitepaper_artifacts,
            missing_tests=[
                "Run graph proximity ablation on dataset-method compatibility queries.",
                "Check for spurious gains from popularity or metadata density.",
            ],
            status=ClaimStatus.NEEDS_VALIDATION,
        ),
    ]


def seed_benchmark_gaps(root: Path | None = None) -> list[BenchmarkGap]:
    """Return initial benchmark gaps."""
    artifacts = existing_inputs(root)
    has_qrels = "artifacts/qrels.jsonl" in artifacts

    return [
        BenchmarkGap(
            gap_id="gap_human_qrels_benchmark",
            title="Human qrels benchmark",
            description="A small expert-labeled relevance set for dataset-method compatibility queries.",
            why_it_matters="Without human qrels, nDCG/MRR can look precise while measuring proxy labels.",
            related_claim_ids=["claim_human_qrels", "claim_dense_semantic_retrieval"],
            expected_artifacts=["artifacts/qrels.jsonl"],
            available_artifacts=["artifacts/qrels.jsonl"] if has_qrels else [],
            blocking_questions=[
                "Which query intents should be labeled first?",
                "What label scale captures compatibility rather than topical similarity?",
            ],
            severity=0.95,
            status=GapStatus.ADDRESSED if has_qrels else GapStatus.OPEN,
        ),
        BenchmarkGap(
            gap_id="gap_hard_negative_violation_tracking",
            title="Hard-negative violation tracking",
            description="A report that detects when near-miss datasets outrank truly compatible datasets.",
            why_it_matters="Scientific search needs to distinguish superficially similar datasets from reusable ones.",
            related_claim_ids=["claim_hard_negatives"],
            expected_artifacts=["reports/field_state/hard_negative_violations.md"],
            available_artifacts=[],
            blocking_questions=[
                "What counts as a violation for method compatibility?",
                "Which negative categories are most damaging?",
            ],
            severity=0.88,
        ),
        BenchmarkGap(
            gap_id="gap_calibration_ece",
            title="Calibration/ECE",
            description="Expected calibration error for recommendation confidence.",
            why_it_matters="Users need to know whether high-confidence recommendations are actually reliable.",
            related_claim_ids=["claim_human_qrels"],
            expected_artifacts=["reports/eval/calibration_ece.json"],
            available_artifacts=[],
            blocking_questions=[
                "What score should be calibrated: relevance, usefulness, or compatibility?",
                "How many labels are needed for a meaningful calibration curve?",
            ],
            severity=0.74,
        ),
        BenchmarkGap(
            gap_id="gap_future_reuse_prediction",
            title="Future reuse prediction",
            description="A benchmark for whether dataset signals predict later reuse or scientific value.",
            why_it_matters="The project should recommend datasets with future validity, not just current discoverability.",
            related_claim_ids=["claim_metadata_richness"],
            expected_artifacts=["reports/field_state/future_reuse_prediction.md"],
            available_artifacts=[],
            blocking_questions=[
                "Which reuse proxy is acceptable for v0.1?",
                "How should age, source, and popularity bias be controlled?",
            ],
            severity=0.72,
        ),
        BenchmarkGap(
            gap_id="gap_analysis_affordance_extraction",
            title="Analysis affordance extraction",
            description="A gold set for whether datasets support specific analyses.",
            why_it_matters="Affordance extraction is central to moving beyond keyword search.",
            related_claim_ids=["claim_affordance_extraction"],
            expected_artifacts=["reports/field_state/analysis_affordance_benchmark.md"],
            available_artifacts=[],
            blocking_questions=[
                "Which affordances matter first for neuroscience dataset reuse?",
                "What evidence is sufficient to mark an affordance as supported?",
            ],
            severity=0.86,
        ),
        BenchmarkGap(
            gap_id="gap_metadata_quality_scoring",
            title="Metadata quality scoring",
            description="A transparent score for completeness, specificity, provenance, and reuse-relevant fields.",
            why_it_matters="Metadata quality may explain both retrieval errors and dataset reuse potential.",
            related_claim_ids=["claim_metadata_richness"],
            expected_artifacts=["reports/field_state/metadata_quality_scoring.md"],
            available_artifacts=[],
            blocking_questions=[
                "Which metadata fields are predictive rather than merely available?",
                "How should missingness be handled across archives?",
            ],
            severity=0.8,
        ),
    ]


def seed_opportunities() -> list[FieldOpportunity]:
    """Return initial opportunities sorted by heuristic score."""
    opportunities = [
        FieldOpportunity(
            opportunity_id="opp_human_qrels_benchmark",
            title="Human qrels benchmark for dataset-method compatibility",
            description="Create a focused expert-labeled benchmark for retrieval and ranking metrics.",
            linked_claim_ids=["claim_human_qrels", "claim_dense_semantic_retrieval"],
            linked_gap_ids=["gap_human_qrels_benchmark"],
            next_step="Draft 20 compatibility queries and label a small candidate pool.",
            novelty_score=7.0,
            feasibility_score=7.0,
            impact_score=10.0,
            uncertainty_reduction_score=10.0,
            personal_fit_score=9.0,
            risk_score=3.0,
            rationale="Highest leverage because it turns retrieval claims into measurable claims.",
        ),
        FieldOpportunity(
            opportunity_id="opp_hard_negative_generator",
            title="Hard-negative generator",
            description="Generate and track near-miss negatives for scientific retrieval queries.",
            linked_claim_ids=["claim_hard_negatives"],
            linked_gap_ids=["gap_hard_negative_violation_tracking"],
            next_step="Define negative categories and produce a small JSONL fixture.",
            novelty_score=6.8,
            feasibility_score=8.2,
            impact_score=8.6,
            uncertainty_reduction_score=8.6,
            personal_fit_score=8.2,
            risk_score=3.0,
            rationale="Practical and directly improves benchmark rigor.",
        ),
        FieldOpportunity(
            opportunity_id="opp_dataset_future_validity_score",
            title="Dataset future validity score",
            description="Estimate whether a dataset is likely to remain reusable and scientifically valuable.",
            linked_claim_ids=["claim_metadata_richness"],
            linked_gap_ids=["gap_future_reuse_prediction"],
            next_step="Define reuse proxies and build a simple retrospective label set.",
            novelty_score=8.6,
            feasibility_score=6.8,
            impact_score=8.6,
            uncertainty_reduction_score=8.2,
            personal_fit_score=8.2,
            risk_score=4.2,
            rationale="High upside, but depends on careful proxy design.",
        ),
        FieldOpportunity(
            opportunity_id="opp_analysis_affordance_benchmark",
            title="Analysis affordance extraction benchmark",
            description="Evaluate whether extracted affordances match what datasets actually support.",
            linked_claim_ids=["claim_affordance_extraction"],
            linked_gap_ids=["gap_analysis_affordance_extraction"],
            next_step="Pick 5 affordances and label 30 dataset-affordance pairs.",
            novelty_score=7.6,
            feasibility_score=6.8,
            impact_score=8.2,
            uncertainty_reduction_score=7.8,
            personal_fit_score=8.1,
            risk_score=4.1,
            rationale="Directly validates the recommendation layer.",
        ),
        FieldOpportunity(
            opportunity_id="opp_provenance_confidence_score",
            title="Provenance confidence score",
            description="Score recommendations by the strength and source of their supporting evidence.",
            linked_claim_ids=["claim_human_qrels"],
            linked_gap_ids=["gap_calibration_ece"],
            next_step="Map evidence sources to confidence priors and audit examples.",
            novelty_score=6.8,
            feasibility_score=7.8,
            impact_score=7.4,
            uncertainty_reduction_score=7.2,
            personal_fit_score=7.7,
            risk_score=3.6,
            rationale="Good bridge between retrieval scores and scientific trust.",
        ),
        FieldOpportunity(
            opportunity_id="opp_metadata_richness_vs_reuse",
            title="Metadata richness vs reuse value study",
            description="Test whether richer metadata predicts reuse-relevant labels or retrieval success.",
            linked_claim_ids=["claim_metadata_richness"],
            linked_gap_ids=["gap_metadata_quality_scoring", "gap_future_reuse_prediction"],
            next_step="Define metadata richness features and correlate against existing readiness signals.",
            novelty_score=6.5,
            feasibility_score=7.5,
            impact_score=7.1,
            uncertainty_reduction_score=7.0,
            personal_fit_score=7.4,
            risk_score=3.8,
            rationale="Useful, scoped, and likely to explain current retrieval failures.",
        ),
        FieldOpportunity(
            opportunity_id="opp_dataset_method_transfer_map",
            title="Dataset-method transfer map",
            description="Map which dataset properties transfer across analysis methods and scientific questions.",
            linked_claim_ids=["claim_graph_proximity", "claim_affordance_extraction"],
            linked_gap_ids=["gap_analysis_affordance_extraction"],
            next_step="Prototype a small map from methods to required data affordances.",
            novelty_score=8.2,
            feasibility_score=5.8,
            impact_score=7.7,
            uncertainty_reduction_score=6.8,
            personal_fit_score=7.6,
            risk_score=5.2,
            rationale="Strategically important but less immediate than benchmark foundations.",
        ),
        FieldOpportunity(
            opportunity_id="opp_uncertainty_aware_recommendation",
            title="Uncertainty-aware dataset recommendation",
            description="Expose uncertainty and missing evidence alongside ranked dataset recommendations.",
            linked_claim_ids=["claim_human_qrels", "claim_metadata_richness"],
            linked_gap_ids=["gap_calibration_ece"],
            next_step="Add a prototype uncertainty explanation to result cards after qrels exist.",
            novelty_score=7.4,
            feasibility_score=5.6,
            impact_score=7.6,
            uncertainty_reduction_score=6.6,
            personal_fit_score=7.2,
            risk_score=5.4,
            rationale="Valuable after calibration and provenance signals mature.",
        ),
    ]
    return rank_opportunities(opportunities)
