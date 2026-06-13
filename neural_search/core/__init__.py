"""Core architecture for the next-generation Neural Search system.

This module provides the foundational components for scientific search:

- claims: Provenance-backed atomic assertions (ReusabilityClaim)
- records: Canonical scientific record model with layered metadata
- dataset_card: DatasetCardV1, CorpusSnapshot, ProvenanceEdge schemas
- query: Query understanding and retrieval planning
- retrieval: Multi-stage retrieval pipeline
- linking: Multi-signal paper-dataset linking

Design Philosophy:
- Provenance-aware: Every label and relationship tracks its source
- Confidence-scored: Quantified uncertainty throughout
- Explainable: Results explain why they matched
- Layered: Raw metadata, normalized, inferred, and learned layers are distinct
"""

from neural_search.core.claims import (
    SOURCE_CONFIDENCE_DEFAULTS,
    ClaimPredicate,
    ClaimStore,
    EvidenceSourceType,
    ReusabilityClaim,
    ReviewStatus,
    claim_has_modality,
    claim_has_task,
    claim_has_variable,
    claim_linked_to_paper,
    claim_supports_affordance,
    create_claim,
    make_claim_id,
)
from neural_search.core.dataset_card import (
    AffordanceRequirement,
    AffordanceValidationResult,
    CorpusSnapshot,
    DatasetCardV1,
    ProvenanceEdge,
    ProvenanceEvidence,
    SourceSnapshot,
    create_corpus_snapshot,
    create_dataset_card_from_normalized,
)
from neural_search.core.neural_signature import (
    CalciumStats,
    FiringRateStats,
    ISIStats,
    NeuralSignatureV1,
    RecordingModality,
    SignatureQuality,
    TrialStats,
    extract_signature_from_metadata,
)
from neural_search.core.query import (
    QueryConstraint,
    QueryIntent,
    QueryPlan,
    RetrievalStage,
    parse_and_plan_query,
)
from neural_search.core.records import (
    EntityType,
    ExtractionProvenance,
    MetadataLayer,
    ScientificEntity,
    ScientificRecord,
    ScientificRecordType,
)
from neural_search.core.retrieval import (
    AffordanceGenerator,
    CandidateSource,
    EmbeddingGenerator,
    LexicalGenerator,
    MultiStageRetriever,
    OntologyGenerator,
    RetrievalResult,
    RetrievalStageResult,
    ScoredCandidate,
    ScoreFuser,
)

__all__ = [
    # Claims (atomic provenance-backed assertions)
    "ClaimPredicate",
    "ClaimStore",
    "EvidenceSourceType",
    "ReusabilityClaim",
    "ReviewStatus",
    "SOURCE_CONFIDENCE_DEFAULTS",
    "claim_has_modality",
    "claim_has_task",
    "claim_has_variable",
    "claim_linked_to_paper",
    "claim_supports_affordance",
    "create_claim",
    "make_claim_id",
    # Dataset Card (new canonical schemas)
    "AffordanceRequirement",
    "AffordanceValidationResult",
    "CorpusSnapshot",
    "DatasetCardV1",
    "ProvenanceEdge",
    "ProvenanceEvidence",
    "SourceSnapshot",
    "create_corpus_snapshot",
    "create_dataset_card_from_normalized",
    # Records
    "EntityType",
    "ExtractionProvenance",
    "MetadataLayer",
    "ScientificEntity",
    "ScientificRecord",
    "ScientificRecordType",
    # Query
    "QueryConstraint",
    "QueryIntent",
    "QueryPlan",
    "RetrievalStage",
    "parse_and_plan_query",
    # Retrieval
    "AffordanceGenerator",
    "CandidateSource",
    "EmbeddingGenerator",
    "LexicalGenerator",
    "MultiStageRetriever",
    "OntologyGenerator",
    "RetrievalResult",
    "RetrievalStageResult",
    "ScoredCandidate",
    "ScoreFuser",
    # Neural Signature
    "CalciumStats",
    "FiringRateStats",
    "ISIStats",
    "NeuralSignatureV1",
    "RecordingModality",
    "SignatureQuality",
    "TrialStats",
    "extract_signature_from_metadata",
]
