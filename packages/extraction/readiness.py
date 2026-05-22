"""Analysis readiness scoring for datasets."""

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import (
    DatasetRecord,
    ExtractionResult,
    AnalysisReadiness,
)


class ReadinessScorer:
    """
    Score dataset readiness for analysis.

    Evaluates metadata completeness, data quality indicators,
    and analysis feasibility.
    """

    # Weight factors for scoring
    WEIGHTS = {
        "has_description": 0.10,
        "has_species": 0.08,
        "has_tasks": 0.12,
        "has_modalities": 0.10,
        "has_brain_regions": 0.08,
        "has_doi": 0.05,
        "has_license": 0.05,
        "has_nwb_files": 0.15,
        "has_multiple_sessions": 0.10,
        "has_trial_structure": 0.12,
        "extraction_confidence": 0.05,
    }

    def score(
        self,
        dataset: DatasetRecord,
        extraction: ExtractionResult | None = None,
    ) -> AnalysisReadiness:
        """
        Calculate analysis readiness score.

        Args:
            dataset: The dataset record.
            extraction: Optional extraction results.

        Returns:
            AnalysisReadiness with score and details.
        """
        scores: dict[str, float] = {}
        strengths: list[str] = []
        limitations: list[str] = []
        missing_metadata: list[str] = []

        # Check description
        if dataset.description and len(dataset.description) > 50:
            scores["has_description"] = 1.0
            strengths.append("Detailed description available")
        else:
            scores["has_description"] = 0.0
            missing_metadata.append("description")

        # Check species
        if dataset.species:
            scores["has_species"] = 1.0
            strengths.append(f"Species identified: {', '.join(dataset.species)}")
        else:
            scores["has_species"] = 0.0
            missing_metadata.append("species")

        # Check tasks
        if dataset.tasks:
            scores["has_tasks"] = 1.0
            strengths.append(f"Tasks identified: {', '.join(dataset.tasks[:3])}")
        elif extraction and extraction.task_labels:
            scores["has_tasks"] = 0.7
            tasks = [l.label for l in extraction.task_labels[:3]]
            strengths.append(f"Tasks extracted: {', '.join(tasks)}")
        else:
            scores["has_tasks"] = 0.0
            limitations.append("No behavioral task identified")
            missing_metadata.append("tasks")

        # Check modalities
        if dataset.modalities:
            scores["has_modalities"] = 1.0
            strengths.append(
                f"Recording modalities: {', '.join(dataset.modalities[:3])}"
            )
        elif extraction and extraction.modality_labels:
            scores["has_modalities"] = 0.7
            mods = [l.label for l in extraction.modality_labels[:3]]
            strengths.append(f"Modalities detected: {', '.join(mods)}")
        else:
            scores["has_modalities"] = 0.0
            missing_metadata.append("modalities")

        # Check brain regions
        if dataset.brain_regions:
            scores["has_brain_regions"] = 1.0
            strengths.append("Brain regions documented")
        else:
            scores["has_brain_regions"] = 0.0
            missing_metadata.append("brain_regions")

        # Check DOI
        if dataset.doi:
            scores["has_doi"] = 1.0
            strengths.append("DOI available for citation")
        else:
            scores["has_doi"] = 0.0
            limitations.append("No DOI for citation")

        # Check license
        if dataset.license:
            scores["has_license"] = 1.0
        else:
            scores["has_license"] = 0.0
            limitations.append("License not specified")

        # Check NWB files
        if dataset.nwb_count > 0:
            scores["has_nwb_files"] = 1.0
            strengths.append(f"{dataset.nwb_count} NWB files available")
        elif dataset.data_standard and dataset.data_standard.value == "bids":
            scores["has_nwb_files"] = 0.8  # BIDS is also good
            strengths.append("BIDS format dataset")
        else:
            scores["has_nwb_files"] = 0.0
            limitations.append("No standardized data files detected")

        # Check multiple sessions (proxy: count assets)
        if len(dataset.assets) > 5:
            scores["has_multiple_sessions"] = 1.0
            strengths.append(f"Multiple sessions ({len(dataset.assets)} files)")
        elif len(dataset.assets) > 1:
            scores["has_multiple_sessions"] = 0.5
        else:
            scores["has_multiple_sessions"] = 0.0
            limitations.append("Limited data (single session)")

        # Trial structure (check description for keywords)
        trial_keywords = ["trial", "trials", "events", "stimulus", "response"]
        desc_lower = (dataset.description or "").lower()
        if any(kw in desc_lower for kw in trial_keywords):
            scores["has_trial_structure"] = 1.0
            strengths.append("Trial structure indicated")
        else:
            scores["has_trial_structure"] = 0.0

        # Extraction confidence
        if extraction and extraction.task_labels:
            avg_confidence = sum(
                l.confidence for l in extraction.task_labels
            ) / len(extraction.task_labels)
            scores["extraction_confidence"] = avg_confidence
        else:
            scores["extraction_confidence"] = 0.0

        # Calculate weighted score
        total_score = sum(
            scores.get(k, 0.0) * v for k, v in self.WEIGHTS.items()
        )

        # Get suggested analyses
        suggested_analyses = self._get_suggested_analyses(dataset, extraction)

        return AnalysisReadiness(
            score=min(total_score, 1.0),
            strengths=strengths,
            limitations=limitations,
            missing_metadata=missing_metadata,
            suggested_analyses=suggested_analyses,
        )

    def _get_suggested_analyses(
        self,
        dataset: DatasetRecord,
        extraction: ExtractionResult | None,
    ) -> list[str]:
        """Get suggested analyses based on dataset characteristics."""
        suggestions = []

        # From extraction
        if extraction and extraction.analysis_affordances:
            suggestions.extend(extraction.analysis_affordances[:5])

        # Generic suggestions based on modalities
        if dataset.modalities:
            mod_set = set(m.lower() for m in dataset.modalities)
            if "extracellular_ephys" in mod_set or "neuropixels" in mod_set:
                suggestions.extend([
                    "spike_sorting",
                    "population_decoding",
                    "cross_correlation",
                ])
            if "calcium_imaging" in mod_set:
                suggestions.extend([
                    "roi_extraction",
                    "activity_correlation",
                    "ensemble_analysis",
                ])
            if "behavior_video" in mod_set:
                suggestions.extend([
                    "pose_estimation",
                    "behavioral_segmentation",
                ])

        # Deduplicate and limit
        seen = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique[:10]
