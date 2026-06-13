"""RAG-grounded neuroscience LLM adjudicator for query-dataset relevance.

Produces evidence-grounded 0–3 qrels labels with full provenance.
Labels are ``neuro_judge`` provenance — NOT human gold.
"""

from neural_search.eval.neuro_judge.calibration import CalibrationReport, calibrate
from neural_search.eval.neuro_judge.consensus import build_consensus
from neural_search.eval.neuro_judge.evidence_packet import (
    NEURO_JUDGE_WATERMARK,
    ConflictRecord,
    ConsensusResult,
    EvidencePacket,
    LabelProvenance,
    NeuroJudgment,
)
from neural_search.eval.neuro_judge.judge import (
    MockNeuroJudge,
    NeuroJudgeProtocol,
    build_neuro_judge,
)

__all__ = [
    "EvidencePacket",
    "NeuroJudgment",
    "ConsensusResult",
    "ConflictRecord",
    "LabelProvenance",
    "NEURO_JUDGE_WATERMARK",
    "NeuroJudgeProtocol",
    "MockNeuroJudge",
    "build_neuro_judge",
    "build_consensus",
    "calibrate",
    "CalibrationReport",
]
