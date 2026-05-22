"""Extract species labels from metadata."""

import re

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import ExtractionLabel


# Species mapping with synonyms
SPECIES_MAP = {
    "mus_musculus": {
        "canonical": "Mus musculus",
        "common": "mouse",
        "patterns": [
            r"\bmouse\b",
            r"\bmice\b",
            r"mus\s*musculus",
            r"\bmurine\b",
        ],
    },
    "rattus_norvegicus": {
        "canonical": "Rattus norvegicus",
        "common": "rat",
        "patterns": [r"\brat\b", r"\brats\b", r"rattus", r"norvegicus"],
    },
    "homo_sapiens": {
        "canonical": "Homo sapiens",
        "common": "human",
        "patterns": [
            r"\bhuman\b",
            r"\bhumans\b",
            r"homo\s*sapiens",
            r"\bsubject\b",
            r"\bparticipant\b",
        ],
    },
    "macaca_mulatta": {
        "canonical": "Macaca mulatta",
        "common": "rhesus macaque",
        "patterns": [
            r"macaque",
            r"macaca",
            r"rhesus",
            r"\bmonkey\b",
            r"\bmonkeys\b",
            r"\bprimate\b",
        ],
    },
    "drosophila_melanogaster": {
        "canonical": "Drosophila melanogaster",
        "common": "fruit fly",
        "patterns": [
            r"drosophila",
            r"fruit\s*fly",
            r"fruitfly",
            r"melanogaster",
        ],
    },
    "danio_rerio": {
        "canonical": "Danio rerio",
        "common": "zebrafish",
        "patterns": [r"zebrafish", r"danio", r"rerio"],
    },
    "caenorhabditis_elegans": {
        "canonical": "Caenorhabditis elegans",
        "common": "C. elegans",
        "patterns": [r"c\.?\s*elegans", r"caenorhabditis", r"\bworm\b"],
    },
}


class SpeciesExtractor:
    """Extract species labels from metadata."""

    def __init__(self):
        self._compiled_patterns = {}
        for species_id, config in SPECIES_MAP.items():
            self._compiled_patterns[species_id] = [
                re.compile(p, re.IGNORECASE) for p in config["patterns"]
            ]

    def extract(
        self,
        text_fields: dict[str, str],
        explicit_species: list[str] | None = None,
    ) -> list[ExtractionLabel]:
        """
        Extract species labels from text and explicit metadata.

        Args:
            text_fields: Dict mapping field names to text content.
            explicit_species: Explicit species values from metadata.

        Returns:
            List of extracted species labels.
        """
        labels: list[ExtractionLabel] = []
        seen_species: set[str] = set()

        # Process explicit species first (higher confidence)
        if explicit_species:
            for species in explicit_species:
                normalized = self._normalize_species(species)
                if normalized and normalized not in seen_species:
                    seen_species.add(normalized)
                    labels.append(
                        ExtractionLabel(
                            label=normalized,
                            confidence=0.95,
                            evidence=f"Explicit metadata: {species}",
                            source_span=species,
                            extractor="explicit_metadata",
                        )
                    )

        # Extract from text
        combined_text = " ".join(text_fields.values())
        for species_id, patterns in self._compiled_patterns.items():
            if species_id in seen_species:
                continue

            for pattern in patterns:
                match = pattern.search(combined_text)
                if match:
                    seen_species.add(species_id)
                    config = SPECIES_MAP[species_id]
                    labels.append(
                        ExtractionLabel(
                            label=species_id,
                            confidence=0.85,
                            evidence=f"Pattern match: {match.group()}",
                            source_span=match.group(),
                            extractor="species_regex",
                        )
                    )
                    break

        return labels

    def _normalize_species(self, species: str) -> str | None:
        """Normalize a species string to canonical form."""
        species_lower = species.lower().strip()

        for species_id, config in SPECIES_MAP.items():
            # Check canonical name
            if config["canonical"].lower() == species_lower:
                return species_id

            # Check common name
            if config["common"].lower() == species_lower:
                return species_id

            # Check patterns
            for pattern in self._compiled_patterns[species_id]:
                if pattern.search(species):
                    return species_id

        # Return original if no match
        return species if species else None
