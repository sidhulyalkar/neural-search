"""Schema for the methodology registry overlay.

Bridges named analytical techniques in ``data/methods/methods_taxonomy.yaml``
(formulas, assumptions, limitations, key papers — see
``neural_search.ingestion.methods_builder``) to the analysis-family
vocabulary already used by ``neural_search.awareness.taxonomy.DATA_FORMS``
and populated into the real corpus graph by
``neural_search.graph.builder.build_taxonomy_requirement_subgraph``.

This does not replace either existing vocabulary — it is a thin index that
lets a technique's depth (formula/assumptions/key papers) be reached from an
``analysis_affordance`` node already present in the production graph.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field, field_validator, model_validator


@lru_cache(maxsize=1)
def known_taxonomy_method_ids() -> frozenset[str]:
    """All method/concept ids declared in data/methods/methods_taxonomy.yaml."""

    from neural_search.ingestion.methods_builder import _load_taxonomy

    taxonomy = _load_taxonomy()
    ids: set[str] = set()
    for category in taxonomy.get("categories", []):
        for item in category.get("methods", category.get("concepts", [])):
            ids.add(item["id"])
    return frozenset(ids)


@lru_cache(maxsize=1)
def known_analysis_families() -> frozenset[str]:
    """All analysis_family strings declared across awareness.taxonomy.DATA_FORMS."""

    from neural_search.awareness.taxonomy import DATA_FORMS

    families: set[str] = set()
    for data_form in DATA_FORMS.values():
        families.update(data_form.analysis_families)
    return frozenset(families)


@lru_cache(maxsize=1)
def known_affordance_ids() -> frozenset[str]:
    """The 18 live per-dataset affordance ids from analysis_affordances.py."""

    from neural_search.analysis_affordances import AFFORDANCE_IDS

    return frozenset(AFFORDANCE_IDS)


@lru_cache(maxsize=1)
def known_ontology_affordance_ids() -> frozenset[str]:
    """Affordance ids declared in data/ontology/behavioral_task_ontology.yaml."""

    from neural_search.ontology.loader import load_ontology

    return frozenset(load_ontology().affordance_ids())


class MethodAnalysisLink(BaseModel):
    """One analysis family's bridge to the named techniques that implement it."""

    analysis_family: str
    taxonomy_method_ids: list[str] = Field(min_length=1)
    rationale: str
    cross_ref_affordance_id: str | None = None
    cross_ref_ontology_affordance_id: str | None = None
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    requires_human_review: bool = False

    @field_validator("analysis_family")
    @classmethod
    def analysis_family_must_be_known(cls, value: str) -> str:
        known = known_analysis_families()
        if value not in known:
            raise ValueError(
                f"analysis_family {value!r} is not declared by any "
                "DataForm.analysis_families in neural_search.awareness.taxonomy"
            )
        return value

    @field_validator("taxonomy_method_ids")
    @classmethod
    def taxonomy_methods_must_be_known(cls, value: list[str]) -> list[str]:
        known = known_taxonomy_method_ids()
        unknown = sorted(m for m in value if m not in known)
        if unknown:
            raise ValueError(
                f"unknown methods_taxonomy.yaml id(s): {unknown}"
            )
        return value

    @field_validator("rationale")
    @classmethod
    def rationale_must_be_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("rationale must not be empty")
        return value

    @field_validator("cross_ref_affordance_id")
    @classmethod
    def cross_ref_affordance_must_be_known(cls, value: str | None) -> str | None:
        if value is not None and value not in known_affordance_ids():
            raise ValueError(
                f"cross_ref_affordance_id {value!r} is not one of "
                "analysis_affordances.AFFORDANCE_IDS"
            )
        return value

    @field_validator("cross_ref_ontology_affordance_id")
    @classmethod
    def cross_ref_ontology_affordance_must_be_known(cls, value: str | None) -> str | None:
        if value is not None and value not in known_ontology_affordance_ids():
            raise ValueError(
                f"cross_ref_ontology_affordance_id {value!r} is not declared in "
                "data/ontology/behavioral_task_ontology.yaml"
            )
        return value


class MethodRegistry(BaseModel):
    """Top-level container loaded from data/methods/method_registry.yaml."""

    version: str = "2.0"
    links: list[MethodAnalysisLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_analysis_families(self) -> "MethodRegistry":
        families = [link.analysis_family for link in self.links]
        dupes = sorted({f for f in families if families.count(f) > 1})
        if dupes:
            raise ValueError(f"duplicate analysis_family entries: {dupes}")
        return self
