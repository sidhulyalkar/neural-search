"""Pydantic models for the behavioral task ontology."""

from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """A behavioral task in the ontology."""

    id: str = Field(..., description="Unique identifier for the task")
    label: str = Field(..., description="Human-readable task name")
    category: str = Field(..., description="Task category (e.g., decision_making)")
    definition: str = Field(..., description="Detailed task definition")
    synonyms: list[str] = Field(default_factory=list, description="Alternative names")
    common_events: list[str] = Field(
        default_factory=list, description="Typical trial events"
    )
    relevant_modalities: list[str] = Field(
        default_factory=list, description="Recording modalities"
    )
    relevant_regions: list[str] = Field(
        default_factory=list, description="Brain regions"
    )
    suggested_analyses: list[str] = Field(
        default_factory=list, description="Recommended analyses"
    )

    def matches(self, query: str) -> bool:
        """Check if query matches this task (case-insensitive)."""
        query_lower = query.lower()
        if query_lower in self.id.lower():
            return True
        if query_lower in self.label.lower():
            return True
        for syn in self.synonyms:
            if query_lower in syn.lower():
                return True
        return False


class BehaviorLabel(BaseModel):
    """A behavior label extracted from metadata."""

    id: str
    label: str
    category: str
    synonyms: list[str] = Field(default_factory=list)


class Modality(BaseModel):
    """A recording modality."""

    id: str
    label: str
    category: str
    synonyms: list[str] = Field(default_factory=list)
    file_extensions: list[str] = Field(default_factory=list)


class BrainRegion(BaseModel):
    """A brain region."""

    id: str
    label: str
    abbreviation: Optional[str] = None
    parent: Optional[str] = None
    synonyms: list[str] = Field(default_factory=list)


class Species(BaseModel):
    """A species."""

    id: str
    label: str
    common_name: str
    synonyms: list[str] = Field(default_factory=list)


class Ontology(BaseModel):
    """Complete ontology containing all taxonomies."""

    tasks: list[Task] = Field(default_factory=list)
    behaviors: list[BehaviorLabel] = Field(default_factory=list)
    modalities: list[Modality] = Field(default_factory=list)
    regions: list[BrainRegion] = Field(default_factory=list)
    species: list[Species] = Field(default_factory=list)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_task_by_label(self, label: str) -> Optional[Task]:
        """Get a task by label (case-insensitive)."""
        label_lower = label.lower()
        for task in self.tasks:
            if task.label.lower() == label_lower:
                return task
        return None

    def search_tasks(self, query: str) -> list[Task]:
        """Search tasks by query string."""
        return [t for t in self.tasks if t.matches(query)]

    def get_categories(self) -> list[str]:
        """Get all unique task categories."""
        return list(set(t.category for t in self.tasks))

    def get_tasks_by_category(self, category: str) -> list[Task]:
        """Get all tasks in a category."""
        return [t for t in self.tasks if t.category == category]

    def get_all_modalities(self) -> set[str]:
        """Get all modalities mentioned across tasks."""
        modalities = set()
        for task in self.tasks:
            modalities.update(task.relevant_modalities)
        return modalities

    def get_all_regions(self) -> set[str]:
        """Get all brain regions mentioned across tasks."""
        regions = set()
        for task in self.tasks:
            regions.update(task.relevant_regions)
        return regions
