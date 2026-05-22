"""Dataset card generator."""

from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import (
    DatasetRecord,
    DatasetCard,
    ExtractionResult,
    AnalysisReadiness,
)
from extraction import MetadataExtractor, ReadinessScorer

from .markdown import MarkdownRenderer


class CardGenerator:
    """
    Generate comprehensive dataset cards.

    Combines metadata, extraction results, and readiness scoring
    into structured cards with both JSON and Markdown output.
    """

    def __init__(self):
        self.extractor = MetadataExtractor()
        self.scorer = ReadinessScorer()
        self.renderer = MarkdownRenderer()

    def generate(
        self,
        dataset: DatasetRecord,
        extraction: Optional[ExtractionResult] = None,
        readiness: Optional[AnalysisReadiness] = None,
    ) -> DatasetCard:
        """
        Generate a dataset card.

        Args:
            dataset: The dataset record.
            extraction: Pre-computed extraction results (will compute if None).
            readiness: Pre-computed readiness score (will compute if None).

        Returns:
            Complete DatasetCard with all metadata and rendered markdown.
        """
        # Run extraction if needed
        if extraction is None:
            extraction = self.extractor.extract(dataset)

        # Score readiness if needed
        if readiness is None:
            readiness = self.scorer.score(dataset, extraction)

        # Build the card
        card = DatasetCard(
            dataset_id=dataset.id,
            title=dataset.title,
            summary=self._generate_summary(dataset, extraction),
            source=dataset.source,
            data_standard=dataset.data_standard or extraction.data_standard,
            species=dataset.species or [
                l.label for l in extraction.species_labels
            ],
            modalities=dataset.modalities or [
                l.label for l in extraction.modality_labels
            ],
            brain_regions=dataset.brain_regions,
            tasks=dataset.tasks or [l.label for l in extraction.task_labels],
            extraction=extraction,
            readiness=readiness,
            url=dataset.url,
            doi=dataset.doi,
            related_papers=[],
            generated_at=datetime.utcnow(),
        )

        # Render markdown
        card.markdown = self.renderer.render(card)

        return card

    def _generate_summary(
        self,
        dataset: DatasetRecord,
        extraction: ExtractionResult,
    ) -> str:
        """Generate a concise summary of the dataset."""
        parts = []

        # Data source
        parts.append(f"Dataset from {dataset.source.value.upper()}")

        # Species
        if dataset.species:
            parts.append(f"using {', '.join(dataset.species[:2])}")

        # Tasks
        tasks = dataset.tasks or [l.label for l in extraction.task_labels[:2]]
        if tasks:
            parts.append(f"studying {', '.join(tasks[:2])}")

        # Modalities
        modalities = dataset.modalities or [
            l.label for l in extraction.modality_labels[:2]
        ]
        if modalities:
            parts.append(f"with {', '.join(modalities[:2])}")

        # File counts
        if dataset.nwb_count > 0:
            parts.append(f"({dataset.nwb_count} NWB files)")

        return " ".join(parts) + "."

    def generate_batch(
        self, datasets: list[DatasetRecord]
    ) -> list[DatasetCard]:
        """Generate cards for multiple datasets."""
        return [self.generate(d) for d in datasets]
