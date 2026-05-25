"""Canonical species and model-organism helpers for neuroscience search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpeciesProfile:
    """Canonical organism metadata used by search, graph, and reports."""

    species_id: str
    label: str
    taxon_id: str | None
    aliases: tuple[str, ...]
    taxon_groups: tuple[str, ...]
    animal_type: str
    model_roles: tuple[str, ...]


SPECIES_PROFILES: dict[str, SpeciesProfile] = {
    "human": SpeciesProfile(
        species_id="human",
        label="Human",
        taxon_id="NCBITaxon:9606",
        aliases=("human", "humans", "participant", "participants", "patient", "patients", "homo sapiens"),
        taxon_groups=("primate", "human"),
        animal_type="human",
        model_roles=("clinical_cohort", "human_subject"),
    ),
    "mouse": SpeciesProfile(
        species_id="mouse",
        label="Mouse",
        taxon_id="NCBITaxon:10090",
        aliases=("mouse", "mice", "mus musculus", "murine", "mouse model"),
        taxon_groups=("rodent", "mammal"),
        animal_type="model_organism",
        model_roles=("genetic_model", "rodent_model"),
    ),
    "rat": SpeciesProfile(
        species_id="rat",
        label="Rat",
        taxon_id="NCBITaxon:10116",
        aliases=("rat", "rats", "rattus", "rattus norvegicus", "rat model"),
        taxon_groups=("rodent", "mammal"),
        animal_type="model_organism",
        model_roles=("behavioral_model", "rodent_model"),
    ),
    "macaque": SpeciesProfile(
        species_id="macaque",
        label="Macaque",
        taxon_id="NCBITaxon:9541",
        aliases=(
            "macaque",
            "macaques",
            "rhesus",
            "rhesus macaque",
            "monkey",
            "nonhuman primate",
            "non-human primate",
            "nhp",
        ),
        taxon_groups=("non_human_primate", "primate", "mammal"),
        animal_type="model_organism",
        model_roles=("large_animal_model", "primate_model"),
    ),
    "marmoset": SpeciesProfile(
        species_id="marmoset",
        label="Marmoset",
        taxon_id="NCBITaxon:9483",
        aliases=("marmoset", "marmosets", "common marmoset", "callithrix jacchus", "nhp"),
        taxon_groups=("non_human_primate", "primate", "mammal"),
        animal_type="model_organism",
        model_roles=("large_animal_model", "primate_model"),
    ),
    "zebrafish": SpeciesProfile(
        species_id="zebrafish",
        label="Zebrafish",
        taxon_id="NCBITaxon:7955",
        aliases=("zebrafish", "danio rerio", "larval zebrafish", "fish"),
        taxon_groups=("fish", "vertebrate"),
        animal_type="model_organism",
        model_roles=("developmental_model", "whole_brain_imaging_model"),
    ),
    "drosophila": SpeciesProfile(
        species_id="drosophila",
        label="Drosophila",
        taxon_id="NCBITaxon:7227",
        aliases=("drosophila", "fruit fly", "fly", "drosophila melanogaster"),
        taxon_groups=("invertebrate", "insect"),
        animal_type="model_organism",
        model_roles=("genetic_model", "compact_circuit_model"),
    ),
    "c_elegans": SpeciesProfile(
        species_id="c_elegans",
        label="C. elegans",
        taxon_id="NCBITaxon:6239",
        aliases=("c elegans", "c. elegans", "caenorhabditis elegans", "worm", "nematode"),
        taxon_groups=("invertebrate", "nematode"),
        animal_type="model_organism",
        model_roles=("connectome_model", "genetic_model"),
    ),
    "ferret": SpeciesProfile(
        species_id="ferret",
        label="Ferret",
        taxon_id="NCBITaxon:9669",
        aliases=("ferret", "ferrets", "mustela putorius furo"),
        taxon_groups=("carnivore", "mammal"),
        animal_type="model_organism",
        model_roles=("sensory_system_model",),
    ),
    "songbird": SpeciesProfile(
        species_id="songbird",
        label="Songbird",
        taxon_id=None,
        aliases=("songbird", "songbirds", "zebra finch", "taeniopygia guttata", "bird", "avian"),
        taxon_groups=("bird", "vertebrate"),
        animal_type="model_organism",
        model_roles=("vocal_learning_model",),
    ),
    "mixed_species": SpeciesProfile(
        species_id="mixed_species",
        label="Mixed species",
        taxon_id=None,
        aliases=("mixed species", "multi species", "multispecies", "cross species", "cross-species"),
        taxon_groups=("mixed_species",),
        animal_type="mixed",
        model_roles=("comparative_neuroscience",),
    ),
}

GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "rodent": ("rodent", "rodents"),
    "non_human_primate": ("non human primate", "non-human primate", "nonhuman primate", "nhp", "monkey", "monkeys"),
    "primate": ("primate", "primates"),
    "mammal": ("mammal", "mammals"),
    "vertebrate": ("vertebrate", "vertebrates"),
    "invertebrate": ("invertebrate", "invertebrates"),
    "fish": ("fish",),
    "bird": ("bird", "birds", "avian"),
    "insect": ("insect", "insects", "fly", "flies"),
    "nematode": ("nematode", "nematodes", "worm", "worms"),
}


def _norm(value: str) -> str:
    cleaned = str(value).casefold().replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


def _snake(value: str) -> str:
    return "_".join(_norm(value).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = _norm(phrase)
    if not normalized:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text) is not None


def _alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for profile in SPECIES_PROFILES.values():
        aliases[_norm(profile.species_id)] = profile.species_id
        aliases[_norm(profile.label)] = profile.species_id
        for alias in profile.aliases:
            aliases[_norm(alias)] = profile.species_id
    return aliases


ALIAS_TO_SPECIES = _alias_map()


def canonical_species_id(value: Any) -> str | None:
    """Return a canonical species ID when a value is recognized."""

    if value is None:
        return None
    normalized = _norm(str(value))
    if not normalized:
        return None
    return ALIAS_TO_SPECIES.get(normalized)


def get_species_profile(value: Any) -> SpeciesProfile | None:
    """Return a species profile for a canonical ID or alias."""

    species_id = canonical_species_id(value) or _snake(str(value))
    return SPECIES_PROFILES.get(species_id)


def species_terms_for_value(value: Any) -> set[str]:
    """Return exact and broader search terms represented by a species value."""

    raw = _norm(str(value)) if value is not None else ""
    terms = {_snake(raw)} if raw else set()
    profile = get_species_profile(value)
    if profile is None:
        return {term for term in terms if term}
    terms.update(
        {
            profile.species_id,
            _snake(profile.label),
            profile.animal_type,
            *profile.taxon_groups,
            *profile.model_roles,
        }
    )
    terms.update(_snake(alias) for alias in profile.aliases)
    if profile.taxon_id:
        terms.add(profile.taxon_id.casefold())
    return {term for term in terms if term}


def species_terms_for_values(values: list[Any] | tuple[Any, ...] | set[Any]) -> set[str]:
    """Return canonical and broader species terms for many values."""

    terms: set[str] = set()
    for value in values:
        terms.update(species_terms_for_value(value))
    return terms


def species_query_matches(query: str) -> list[dict[str, Any]]:
    """Find exact species or broader animal-type mentions in a query."""

    normalized = _norm(query)
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for profile in SPECIES_PROFILES.values():
        for alias in (profile.species_id, profile.label, *profile.aliases):
            if _contains_phrase(normalized, alias):
                key = (profile.species_id, "exact")
                if key not in seen:
                    seen.add(key)
                    matches.append(
                        {
                            "id": profile.species_id,
                            "label": profile.label,
                            "confidence": 0.96,
                            "evidence": alias,
                            "match_type": "species_alias",
                            "specificity": "exact_species",
                            "taxon_groups": list(profile.taxon_groups),
                            "animal_type": profile.animal_type,
                        }
                    )
                break

    for group_id, aliases in GROUP_ALIASES.items():
        for alias in aliases:
            if _contains_phrase(normalized, alias):
                key = (group_id, "broader")
                if key not in seen:
                    seen.add(key)
                    matches.append(
                        {
                            "id": group_id,
                            "label": group_id.replace("_", " ").title(),
                            "confidence": 0.88,
                            "evidence": alias,
                            "match_type": "taxon_group_alias",
                            "specificity": "broader_taxon",
                            "taxon_groups": [group_id],
                            "animal_type": "taxon_group",
                        }
                    )
                break
    return sorted(matches, key=lambda item: (item["specificity"], item["id"]))


def species_exclusions_for_only_query(query: str) -> list[str]:
    """Return hard-negative species implied by phrases such as ``human only``."""

    normalized = _norm(query)
    allowed: set[str] = set()
    for match in re.finditer(r"\b(?P<species>[a-z0-9 ._-]+?)\s+only\b", normalized):
        species_id = canonical_species_id(match.group("species").strip())
        if species_id:
            allowed.add(species_id)
    if not allowed:
        return []
    return sorted(species_id for species_id in SPECIES_PROFILES if species_id not in allowed)


def species_vocab_aliases() -> dict[str, tuple[str, ...]]:
    """Return aliases keyed by canonical species and broader taxon group IDs."""

    aliases: dict[str, tuple[str, ...]] = {
        profile.species_id: (profile.species_id, profile.label, *profile.aliases)
        for profile in SPECIES_PROFILES.values()
    }
    aliases.update(GROUP_ALIASES)
    return aliases
