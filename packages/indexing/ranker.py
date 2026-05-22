"""Hybrid ranking for search results."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RankingWeights:
    """Weights for different ranking signals."""

    keyword: float = 0.3
    ontology: float = 0.25
    vector: float = 0.25
    readiness: float = 0.1
    recency: float = 0.1


class HybridRanker:
    """
    Combine multiple ranking signals into a final score.

    Signals:
    - Keyword match score
    - Ontology match score
    - Vector similarity score
    - Analysis readiness score
    - Recency score
    """

    def __init__(self, weights: Optional[RankingWeights] = None):
        self.weights = weights or RankingWeights()

    def rank(
        self,
        keyword_score: float = 0.0,
        ontology_score: float = 0.0,
        vector_score: float = 0.0,
        readiness_score: float = 0.0,
        recency_score: float = 0.0,
    ) -> float:
        """
        Compute final ranking score.

        All input scores should be normalized to [0, 1].
        """
        total = (
            self.weights.keyword * keyword_score
            + self.weights.ontology * ontology_score
            + self.weights.vector * vector_score
            + self.weights.readiness * readiness_score
            + self.weights.recency * recency_score
        )

        # Normalize to [0, 1]
        max_possible = (
            self.weights.keyword
            + self.weights.ontology
            + self.weights.vector
            + self.weights.readiness
            + self.weights.recency
        )

        return total / max_possible if max_possible > 0 else 0.0

    def explain(
        self,
        keyword_score: float = 0.0,
        ontology_score: float = 0.0,
        vector_score: float = 0.0,
        readiness_score: float = 0.0,
        recency_score: float = 0.0,
    ) -> dict[str, float]:
        """Return breakdown of score contributions."""
        return {
            "keyword_contribution": self.weights.keyword * keyword_score,
            "ontology_contribution": self.weights.ontology * ontology_score,
            "vector_contribution": self.weights.vector * vector_score,
            "readiness_contribution": self.weights.readiness * readiness_score,
            "recency_contribution": self.weights.recency * recency_score,
        }
