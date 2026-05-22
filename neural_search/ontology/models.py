"""Validated ontology models."""

from __future__ import annotations

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


class Ontology(BaseModel):
    """Complete behavioral ontology loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[Task]
    behavior_labels: list[BehaviorLabel] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "Ontology":
        task_ids = [task.id for task in self.tasks]
        behavior_ids = [behavior.id for behavior in self.behavior_labels]
        duplicate_tasks = {item for item in task_ids if task_ids.count(item) > 1}
        duplicate_behaviors = {
            item for item in behavior_ids if behavior_ids.count(item) > 1
        }
        if duplicate_tasks:
            raise ValueError(f"duplicate task ids: {sorted(duplicate_tasks)}")
        if duplicate_behaviors:
            raise ValueError(f"duplicate behavior ids: {sorted(duplicate_behaviors)}")
        return self

    @property
    def task_by_id(self) -> dict[str, Task]:
        return {task.id: task for task in self.tasks}

    @property
    def behavior_by_id(self) -> dict[str, BehaviorLabel]:
        return {behavior.id: behavior for behavior in self.behavior_labels}

    @property
    def modality_names(self) -> list[str]:
        values = {value for task in self.tasks for value in task.relevant_modalities}
        return sorted(values)

    @property
    def region_names(self) -> list[str]:
        values = {value for task in self.tasks for value in task.relevant_regions}
        return sorted(values)
