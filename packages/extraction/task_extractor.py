"""Extract task labels from metadata using ontology matching."""

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import ExtractionLabel
from ontology import OntologyMatcher


class TaskExtractor:
    """
    Extract behavioral task labels from dataset metadata.

    Uses the ontology matcher to identify tasks from text.
    """

    def __init__(self, matcher: OntologyMatcher):
        self.matcher = matcher

    def extract(self, text_fields: dict[str, str]) -> list[ExtractionLabel]:
        """
        Extract task labels from text fields.

        Args:
            text_fields: Dict mapping field names to text content.

        Returns:
            List of extracted task labels with confidence and evidence.
        """
        labels: list[ExtractionLabel] = []
        seen_task_ids: set[str] = set()

        # Process each text field
        for field_name, text in text_fields.items():
            if not text:
                continue

            # Find task mentions in text
            matches = self.matcher.extract_tasks_from_text(text)

            for match in matches:
                if match.task.id in seen_task_ids:
                    continue

                seen_task_ids.add(match.task.id)

                # Create extraction label
                labels.append(
                    ExtractionLabel(
                        label=match.task.id,
                        confidence=match.confidence,
                        evidence=f"Found in {field_name}: {match.evidence}",
                        source_span=match.matched_term,
                        extractor="ontology_matcher",
                    )
                )

        # Sort by confidence
        labels.sort(key=lambda x: x.confidence, reverse=True)
        return labels

    def extract_from_keywords(
        self, keywords: list[str]
    ) -> list[ExtractionLabel]:
        """Extract tasks from explicit keyword list."""
        labels: list[ExtractionLabel] = []
        seen_task_ids: set[str] = set()

        for keyword in keywords:
            match = self.matcher.match_task(keyword)
            if match and match.task.id not in seen_task_ids:
                seen_task_ids.add(match.task.id)
                labels.append(
                    ExtractionLabel(
                        label=match.task.id,
                        confidence=match.confidence,
                        evidence=f"Matched keyword: {keyword}",
                        source_span=keyword,
                        extractor="ontology_matcher",
                    )
                )

        return labels
