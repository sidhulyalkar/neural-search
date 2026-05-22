"""Extract recording modality labels from metadata."""

import re
from typing import Optional

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import ExtractionLabel, DatasetAsset


# Modality patterns with synonyms
MODALITY_PATTERNS = {
    "extracellular_ephys": {
        "patterns": [
            r"extracellular\s*(electrophysiology|ephys|recording)",
            r"single.?unit",
            r"multi.?unit",
            r"spike\s*sorting",
            r"tetrode",
            r"silicon\s*probe",
        ],
        "keywords": [
            "extracellular",
            "ephys",
            "electrophysiology",
            "spikes",
            "units",
        ],
    },
    "neuropixels": {
        "patterns": [r"neuropixels?", r"npix"],
        "keywords": ["neuropixels", "neuropixel"],
    },
    "calcium_imaging": {
        "patterns": [
            r"calcium\s*imaging",
            r"two.?photon",
            r"2.?photon",
            r"miniscope",
            r"gcamp",
            r"ca2?\+?\s*imaging",
        ],
        "keywords": [
            "calcium imaging",
            "two-photon",
            "miniscope",
            "gcamp",
            "2p imaging",
        ],
    },
    "fiber_photometry": {
        "patterns": [r"fiber\s*photometry", r"fibre\s*photometry", r"photometry"],
        "keywords": ["fiber photometry", "photometry"],
    },
    "eeg": {
        "patterns": [r"\beeg\b", r"electroencephalograph"],
        "keywords": ["eeg", "electroencephalography"],
    },
    "ecog": {
        "patterns": [r"\becog\b", r"electrocorticograph"],
        "keywords": ["ecog", "electrocorticography"],
    },
    "lfp": {
        "patterns": [r"\blfp\b", r"local\s*field\s*potential"],
        "keywords": ["lfp", "local field potential"],
    },
    "fmri": {
        "patterns": [r"\bfmri\b", r"functional\s*mri", r"bold"],
        "keywords": ["fmri", "functional mri", "bold"],
    },
    "behavior_video": {
        "patterns": [
            r"behavior\s*video",
            r"behaviour\s*video",
            r"video\s*tracking",
            r"pose\s*estimation",
            r"deeplabcut",
            r"dlc",
        ],
        "keywords": ["behavior video", "video tracking", "pose estimation"],
    },
    "pupillometry": {
        "patterns": [r"pupil", r"pupillometry", r"eye\s*tracking"],
        "keywords": ["pupil", "pupillometry", "eye tracking"],
    },
    "optogenetics": {
        "patterns": [r"optogenetic", r"channelrhodopsin", r"chr2", r"opto"],
        "keywords": ["optogenetics", "optogenetic", "channelrhodopsin"],
    },
}


class ModalityExtractor:
    """Extract recording modality labels from metadata."""

    def __init__(self):
        # Compile regex patterns
        self._compiled_patterns = {}
        for modality, config in MODALITY_PATTERNS.items():
            self._compiled_patterns[modality] = [
                re.compile(p, re.IGNORECASE) for p in config["patterns"]
            ]

    def extract(
        self,
        text_fields: dict[str, str],
        assets: Optional[list[DatasetAsset]] = None,
    ) -> list[ExtractionLabel]:
        """
        Extract modality labels from text and asset information.

        Args:
            text_fields: Dict mapping field names to text content.
            assets: Optional list of dataset assets.

        Returns:
            List of extracted modality labels.
        """
        labels: list[ExtractionLabel] = []
        seen_modalities: set[str] = set()

        # Extract from text
        combined_text = " ".join(text_fields.values())
        text_labels = self._extract_from_text(combined_text)

        for modality, evidence in text_labels:
            if modality not in seen_modalities:
                seen_modalities.add(modality)
                labels.append(
                    ExtractionLabel(
                        label=modality,
                        confidence=0.9,
                        evidence=f"Pattern match: {evidence}",
                        source_span=evidence,
                        extractor="modality_regex",
                    )
                )

        # Extract from assets
        if assets:
            asset_labels = self._extract_from_assets(assets)
            for modality, evidence in asset_labels:
                if modality not in seen_modalities:
                    seen_modalities.add(modality)
                    labels.append(
                        ExtractionLabel(
                            label=modality,
                            confidence=0.85,
                            evidence=f"Asset pattern: {evidence}",
                            source_span=evidence,
                            extractor="asset_analysis",
                        )
                    )

        return labels

    def _extract_from_text(self, text: str) -> list[tuple[str, str]]:
        """Extract modalities from text using regex patterns."""
        results = []

        for modality, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    results.append((modality, match.group()))
                    break  # Found this modality, move to next

        return results

    def _extract_from_assets(
        self, assets: list[DatasetAsset]
    ) -> list[tuple[str, str]]:
        """Extract modalities from asset file paths."""
        results = []
        seen = set()

        for asset in assets:
            path_lower = asset.path.lower()

            # Check for specific file patterns
            if any(
                x in path_lower
                for x in ["ecephys", "ephys", "units", "spikes"]
            ):
                if "extracellular_ephys" not in seen:
                    seen.add("extracellular_ephys")
                    results.append(("extracellular_ephys", asset.path))

            if "ophys" in path_lower or "imaging" in path_lower:
                if "calcium_imaging" not in seen:
                    seen.add("calcium_imaging")
                    results.append(("calcium_imaging", asset.path))

            if "behavior" in path_lower or "behaviour" in path_lower:
                if "behavior_video" not in seen:
                    seen.add("behavior_video")
                    results.append(("behavior_video", asset.path))

        return results
