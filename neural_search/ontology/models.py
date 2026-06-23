"""Validated ontology models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REQUIRED_TASK_FIELDS = {
    "id",
    "label",
    "category",
    "definition",
}


class LabelMatch(BaseModel):
    """Evidence-bearing match emitted by the ontology matcher."""

    id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    category: str | None = None
    match_type: str = "exact"


class Task(BaseModel):
    """Behavioral task schema. Every field is required in the YAML."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    category: str
    definition: str
    synonyms: list[str] = Field(default_factory=list)
    common_events: list[str] = Field(default_factory=list)
    relevant_modalities: list[str] = Field(default_factory=list)
    relevant_regions: list[str] = Field(default_factory=list)
    suggested_analyses: list[str] = Field(default_factory=list)

    @field_validator("id", "label", "category", "definition")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator(
        "synonyms",
        "common_events",
        "relevant_modalities",
        "relevant_regions",
        "suggested_analyses",
    )
    @classmethod
    def non_empty_string_list(cls, values: list[str]) -> list[str]:
        if not isinstance(values, list):
            raise ValueError("must be a list")
        cleaned = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError("all entries must be strings")
            value = value.strip()
            if not value:
                raise ValueError("entries must not be empty")
            cleaned.append(value)
        return cleaned


class BehaviorLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    synonyms: list[str]

    @field_validator("id", "label")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("synonyms")
    @classmethod
    def non_empty_synonyms(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("must include at least one synonym")
        return [value.strip() for value in values]


class AnalysisAffordance(BaseModel):
    """Analysis affordance schema for what analyses a dataset can support."""

    model_config = ConfigDict(extra="allow")

    id: str
    label: str
    definition: str = ""
    required_signals: list[str] = Field(default_factory=list)
    helpful_signals: list[str] = Field(default_factory=list)
    typical_outputs: list[str] = Field(default_factory=list)
    relevant_tasks: list[str] = Field(default_factory=list)
    query_synonyms: list[str] = Field(default_factory=list)

    @field_validator("id", "label")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class BrainRegion(BaseModel):
    """Search-oriented brain region entry with aliases and parent links."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    system: str = ""
    species_scope: list[str] = Field(default_factory=lambda: ["cross_species"])
    atlas_refs: dict[str, str] = Field(default_factory=dict)
    species_aliases: dict[str, list[str]] = Field(default_factory=dict)
    disambiguation_notes: list[str] = Field(default_factory=list)
    strict: bool = False

    @field_validator("id", "label")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("aliases", "parents", "children", "species_scope", "disambiguation_notes")
    @classmethod
    def clean_string_list(cls, values: list[str]) -> list[str]:
        if not isinstance(values, list):
            raise ValueError("must be a list")
        cleaned = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError("all entries must be strings")
            value = value.strip()
            if value:
                cleaned.append(value)
        return cleaned

    @field_validator("system")
    @classmethod
    def clean_optional_string(cls, value: str) -> str:
        return value.strip()

    @field_validator("atlas_refs", mode="before")
    @classmethod
    def clean_atlas_refs(cls, values: dict[str, Any]) -> dict[str, str]:
        if not isinstance(values, dict):
            raise ValueError("must be a mapping")
        return {
            str(key).strip(): str(value).strip()
            for key, value in values.items()
            if value is not None and str(key).strip() and str(value).strip()
        }

    @field_validator("species_aliases")
    @classmethod
    def clean_species_aliases(cls, values: dict[str, list[str]]) -> dict[str, list[str]]:
        if not isinstance(values, dict):
            raise ValueError("must be a mapping")
        cleaned: dict[str, list[str]] = {}
        for species, aliases in values.items():
            if not isinstance(aliases, list):
                raise ValueError("species_aliases values must be lists")
            cleaned_aliases = [str(alias).strip() for alias in aliases if str(alias).strip()]
            if cleaned_aliases:
                cleaned[str(species).strip()] = cleaned_aliases
        return cleaned


class RecordingScale(BaseModel):
    """How a dataset samples neural information below broad modality level."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    category: str
    sampling_unit: str
    signal_form: str
    temporal_resolution: str
    spatial_resolution: str
    compatible_modalities: list[str] = Field(default_factory=list)
    data_types: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    relationship_tags: list[str] = Field(default_factory=list)

    @field_validator(
        "id",
        "label",
        "category",
        "sampling_unit",
        "signal_form",
        "temporal_resolution",
        "spatial_resolution",
    )
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator(
        "compatible_modalities",
        "data_types",
        "aliases",
        "relationship_tags",
    )
    @classmethod
    def clean_string_list(cls, values: list[str]) -> list[str]:
        if not isinstance(values, list):
            raise ValueError("must be a list")
        cleaned = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError("all entries must be strings")
            value = value.strip()
            if value:
                cleaned.append(value)
        return cleaned


class Ontology(BaseModel):
    """Complete behavioral ontology loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[Task]
    behavior_labels: list[BehaviorLabel] = Field(default_factory=list)
    analysis_affordances: list[AnalysisAffordance] = Field(default_factory=list)

    @field_validator("analysis_affordances", mode="before")
    @classmethod
    def parse_affordances(cls, values: list[Any]) -> list[AnalysisAffordance]:
        """Parse affordances from dict or model instances."""
        if not values:
            return []
        parsed = []
        for value in values:
            if isinstance(value, AnalysisAffordance):
                parsed.append(value)
            elif isinstance(value, dict):
                parsed.append(AnalysisAffordance.model_validate(value))
        return parsed

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Ontology:
        task_ids = [task.id for task in self.tasks]
        behavior_ids = [behavior.id for behavior in self.behavior_labels]
        affordance_ids = [aff.id for aff in self.analysis_affordances]
        duplicate_tasks = {item for item in task_ids if task_ids.count(item) > 1}
        duplicate_behaviors = {
            item for item in behavior_ids if behavior_ids.count(item) > 1
        }
        duplicate_affordances = {
            item for item in affordance_ids if affordance_ids.count(item) > 1
        }
        if duplicate_tasks:
            raise ValueError(f"duplicate task ids: {sorted(duplicate_tasks)}")
        if duplicate_behaviors:
            raise ValueError(f"duplicate behavior ids: {sorted(duplicate_behaviors)}")
        if duplicate_affordances:
            raise ValueError(f"duplicate affordance ids: {sorted(duplicate_affordances)}")
        return self

    @property
    def task_by_id(self) -> dict[str, Task]:
        return {task.id: task for task in self.tasks}

    @property
    def behavior_by_id(self) -> dict[str, BehaviorLabel]:
        return {behavior.id: behavior for behavior in self.behavior_labels}

    @property
    def affordance_by_id(self) -> dict[str, AnalysisAffordance]:
        return {aff.id: aff for aff in self.analysis_affordances}

    @property
    def modality_names(self) -> list[str]:
        values = {value for task in self.tasks for value in task.relevant_modalities}
        return sorted(values)

    @property
    def region_names(self) -> list[str]:
        values = {value for task in self.tasks for value in task.relevant_regions}
        return sorted(values)

    @property
    def affordance_names(self) -> list[str]:
        return sorted({aff.id for aff in self.analysis_affordances})
