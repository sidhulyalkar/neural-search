"""RAG-grounded neuroscience LLM adjudicator for query-dataset relevance.

Produces evidence-grounded 0–3 qrels labels with full provenance.
Labels are ``neuro_judge`` provenance — NOT human gold.
"""

from neural_search.eval.neuro_judge.evidence_packet import (
    EvidencePacket,
    NeuroJudgment,
    ConsensusResult,
    ConflictRecord,
    LabelProvenance,
    NEURO_JUDGE_WATERMARK,
)
from neural_search.eval.neuro_judge.judge import (
    NeuroJudgeProtocol,
    MockNeuroJudge,
    build_neuro_judge,
)
from neural_search.eval.neuro_judge.consensus import build_consensus
from neural_search.eval.neuro_judge.calibration import calibrate, CalibrationReport

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
