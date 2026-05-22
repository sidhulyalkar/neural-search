"""Validated ontology loading and fuzzy matching."""

from neural_search.ontology.loader import (
    DEFAULT_ONTOLOGY_PATH,
    OntologyValidationError,
    get_all_tasks,
    get_ontology,
    get_task_by_id,
    load_ontology,
    reload_ontology,
    validate_ontology,
)
from neural_search.ontology.matcher import (
    OntologyMatcher,
    expand_query_terms,
    match_all,
    match_behavior_labels,
    match_brain_regions,
    match_modalities,
    match_tasks,
    normalize_text,
)
from neural_search.ontology.models import BehaviorLabel, LabelMatch, Ontology, Task

__all__ = [
    "BehaviorLabel",
    "DEFAULT_ONTOLOGY_PATH",
    "LabelMatch",
    "Ontology",
    "OntologyMatcher",
    "OntologyValidationError",
    "Task",
    "expand_query_terms",
    "get_all_tasks",
    "get_ontology",
    "get_task_by_id",
    "load_ontology",
    "match_all",
    "match_behavior_labels",
    "match_brain_regions",
    "match_modalities",
    "match_tasks",
    "normalize_text",
    "reload_ontology",
    "validate_ontology",
]

