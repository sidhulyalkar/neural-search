"""Entity resolution helpers for corpus and graph hardening."""

from neural_search.resolve.entities import (
    EntityConflict,
    EntityResolutionReport,
    EntityResolutionResult,
    canonical_entity_key,
    identifier_keys_for_record,
    resolve_entities,
)

__all__ = [
    "EntityConflict",
    "EntityResolutionReport",
    "EntityResolutionResult",
    "canonical_entity_key",
    "identifier_keys_for_record",
    "resolve_entities",
]
