"""Main metadata extractor combining all sub-extractors."""

from typing import Any, Optional

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import (
    DatasetRecord,
    ExtractionResult,
    ExtractionLabel,
    DataStandard,
)
from ontology import OntologyMatcher, get_ontology

from .task_extractor import TaskExtractor
from .modality_extractor import ModalityExtractor
from .species_extractor import SpeciesExtractor


class MetadataExtractor:
    """
    Extract structured labels from dataset metadata.

    Uses deterministic extraction with ontology matching.
    All extractions include confidence scores and evidence.
    """

    def __init__(self):
        self.ontology = get_ontology()
        self.matcher = OntologyMatcher(self.ontology)
        self.task_extractor = TaskExtractor(self.matcher)
        self.modality_extractor = ModalityExtractor()
        self.species_extractor = SpeciesExtractor()

    def extract(self, dataset: DatasetRecord) -> ExtractionResult:
        """
        Extract all labels from a dataset record.

        Args:
            dataset: The dataset to extract from.

        Returns:
            ExtractionResult with all extracted labels.
        """
        # Combine text fields for extraction
        text_fields = self._get_text_fields(dataset)

        # Extract task labels
        task_labels = self.task_extractor.extract(text_fields)

        # Extract modality labels
        modality_labels = self.modality_extractor.extract(
            text_fields, dataset.assets
        )

        # Extract species labels
        species_labels = self.species_extractor.extract(
            text_fields, dataset.species
        )

        # Detect data standard
        data_standard = self._detect_data_standard(dataset)

        # Determine analysis affordances based on extracted labels
        analysis_affordances = self._get_analysis_affordances(task_labels)

        # Identify missing fields
        missing_fields = self._identify_missing_fields(dataset)

        return ExtractionResult(
            dataset_id=dataset.id,
            task_labels=task_labels,
            behavior_labels=[],  # TODO: Add behavior extraction
            modality_labels=modality_labels,
            region_labels=[],  # TODO: Add region extraction
            species_labels=species_labels,
            data_standard=data_standard,
            analysis_affordances=analysis_affordances,
            missing_fields=missing_fields,
        )

    def _get_text_fields(self, dataset: DatasetRecord) -> dict[str, str]:
        """Extract text fields from dataset for analysis."""
        fields = {
            "title": dataset.title or "",
            "description": dataset.description or "",
        }

        # Add raw metadata fields if available
        if dataset.raw_metadata:
            for key in ["abstract", "keywords", "methods", "protocol"]:
                if key in dataset.raw_metadata:
                    fields[key] = str(dataset.raw_metadata[key])

        return fields

    def _detect_data_standard(
        self, dataset: DatasetRecord
    ) -> Optional[DataStandard]:
        """Detect the data standard used."""
        # Check explicit field
        if dataset.data_standard:
            return dataset.data_standard

        # Check source
        if dataset.source.value == "dandi":
            return DataStandard.NWB
        if dataset.source.value == "openneuro":
            return DataStandard.BIDS

        # Check assets
        for asset in dataset.assets:
            if asset.is_nwb or asset.path.endswith(".nwb"):
                return DataStandard.NWB
            if asset.is_bids or "dataset_description.json" in asset.path:
                return DataStandard.BIDS

        return None

    def _get_analysis_affordances(
        self, task_labels: list[ExtractionLabel]
    ) -> list[str]:
        """Get suggested analyses based on extracted tasks."""
        affordances = set()

        for label in task_labels:
            task = self.ontology.get_task(label.label)
            if task:
                affordances.update(task.suggested_analyses)

        return sorted(affordances)

    def _identify_missing_fields(
        self, dataset: DatasetRecord
    ) -> list[str]:
        """Identify important missing metadata fields."""
        missing = []

        if not dataset.description:
            missing.append("description")
        if not dataset.species:
            missing.append("species")
        if not dataset.doi:
            missing.append("doi")
        if not dataset.license:
            missing.append("license")
        if not dataset.modalities:
            missing.append("modalities")
        if not dataset.tasks:
            missing.append("tasks")
        if not dataset.brain_regions:
            missing.append("brain_regions")

        return missing
