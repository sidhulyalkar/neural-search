"""
Neural Search Ontology Package

Provides experiment-aware ontology for neural data discovery:
- Task taxonomy with synonyms
- Behavior labels
- Modality labels
- Brain region labels
- Analysis suggestions
"""

from .models import Task, BehaviorLabel, Ontology
from .loader import load_ontology, get_ontology
from .matcher import OntologyMatcher

__all__ = [
    "Task",
    "BehaviorLabel",
    "Ontology",
    "load_ontology",
    "get_ontology",
    "OntologyMatcher",
]
