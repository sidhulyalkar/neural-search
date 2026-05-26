"""Deterministic entity resolution for normalized datasets and papers."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from neural_search.normalized import make_dataset_id, make_paper_id

EntityKind = Literal["dataset", "paper", "unknown"]

DOI_URL_RE = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/", flags=re.IGNORECASE)
OPENALEX_URL_RE = re.compile(r"^(?:https?://)?openalex\.org/", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[^a-z0-9._:/-]+")


@dataclass(frozen=True)
class EntityConflict:
    """A deterministic conflict found while merging duplicate records."""

    field: str
    values: tuple[str, ...]
    record_ids: tuple[str, ...]
    resolution: str


@dataclass(frozen=True)
class EntityResolutionResult:
    """One canonical entity plus all observed identifiers and conflicts."""

    canonical_id: str
    entity_type: EntityKind
    record_ids: tuple[str, ...]
    aliases: tuple[str, ...]
    identifiers: dict[str, tuple[str, ...]]
    conflicts: tuple[EntityConflict, ...] = ()


@dataclass(frozen=True)
class EntityResolutionReport:
    """Complete entity resolution output."""

    entities: tuple[EntityResolutionResult, ...]
    lookup: dict[str, str]
    duplicate_groups: tuple[tuple[str, ...], ...]
    conflicts: tuple[EntityConflict, ...] = ()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        path: list[str] = []
        while self.parent[item] != item:
            path.append(item)
            item = self.parent[item]
        for child in path:
            self.parent[child] = item
        return item

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        first, second = sorted([root_left, root_right])
        self.parent[second] = first


def _record_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json", exclude_none=True)
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError(f"record must be a mapping or Pydantic model, got {type(record)!r}")


def _clean_token(value: Any) -> str:
    cleaned = str(value or "").strip().casefold()
    cleaned = TOKEN_RE.sub("_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def normalize_doi(value: Any) -> str:
    """Normalize DOI strings and DOI URLs into a stable key suffix."""

    cleaned = str(value or "").strip()
    cleaned = DOI_URL_RE.sub("", cleaned)
    cleaned = cleaned.removeprefix("doi:").strip()
    cleaned = cleaned.casefold()
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.rstrip(".,;")


def normalize_openalex_id(value: Any) -> str:
    """Normalize OpenAlex IDs or URLs into an uppercase work ID suffix."""

    cleaned = str(value or "").strip()
    cleaned = OPENALEX_URL_RE.sub("", cleaned)
    cleaned = cleaned.removeprefix("openalex:").strip("/")
    if cleaned.casefold().startswith("works/"):
        cleaned = cleaned.split("/", 1)[1]
    return cleaned.upper()


def _entity_type(payload: Mapping[str, Any]) -> EntityKind:
    if payload.get("dataset_id"):
        return "dataset"
    if payload.get("paper_id") or payload.get("doi") or payload.get("authors"):
        return "paper"
    if payload.get("source") and payload.get("source_id"):
        source = _clean_token(payload["source"])
        if source in {"openalex", "pubmed", "crossref", "doi"}:
            return "paper"
        return "dataset"
    return "unknown"


def _add_identifier(
    identifiers: dict[str, set[str]],
    key: str,
    value: Any,
) -> None:
    cleaned = str(value or "").strip()
    if cleaned:
        identifiers[key].add(cleaned)


def identifier_keys_for_record(record: Any) -> dict[str, tuple[str, ...]]:
    """Return normalized identifier keys for a dataset or paper record.

    Keys are prefixed by identifier family, for example ``dataset_id:...``,
    ``source:dandi:000026``, ``doi:10.123/example``, or ``openalex:W123``.
    """

    payload = _record_dict(record)
    entity_type = _entity_type(payload)
    identifiers: dict[str, set[str]] = defaultdict(set)

    if entity_type == "dataset":
        dataset_id = str(payload.get("dataset_id") or "").strip()
        source = str(payload.get("source") or "").strip()
        source_id = str(payload.get("source_id") or "").strip()
        if dataset_id:
            _add_identifier(identifiers, "dataset_id", dataset_id)
        if source and source_id:
            stable_id = make_dataset_id(source, source_id)
            _add_identifier(identifiers, "dataset_id", stable_id)
            _add_identifier(identifiers, "source", f"{_clean_token(source)}:{source_id}")
        for alias in payload.get("aliases", []) or []:
            _add_identifier(identifiers, "alias", _clean_token(alias))

    elif entity_type == "paper":
        paper_id = str(payload.get("paper_id") or "").strip()
        source = str(payload.get("source") or "").strip()
        source_id = str(payload.get("source_id") or "").strip()
        doi = normalize_doi(payload.get("doi"))
        if paper_id:
            _add_identifier(identifiers, "paper_id", paper_id)
            if paper_id.casefold().startswith("paper:openalex:"):
                _add_identifier(identifiers, "openalex", normalize_openalex_id(paper_id.rsplit(":", 1)[-1]))
        if source and source_id:
            stable_id = make_paper_id(source, source_id)
            _add_identifier(identifiers, "paper_id", stable_id)
            _add_identifier(identifiers, "source", f"{_clean_token(source)}:{source_id}")
            if _clean_token(source) == "openalex":
                _add_identifier(identifiers, "openalex", normalize_openalex_id(source_id))
        if doi:
            _add_identifier(identifiers, "doi", doi)
        for alias in payload.get("aliases", []) or []:
            _add_identifier(identifiers, "alias", _clean_token(alias))

    else:
        fallback = str(payload.get("id") or payload.get("source_id") or payload).strip()
        _add_identifier(identifiers, "unknown", _clean_token(fallback))

    return {
        family: tuple(sorted(values))
        for family, values in sorted(identifiers.items())
        if values
    }


def _flatten_identifier_keys(identifiers: Mapping[str, Iterable[str]]) -> tuple[str, ...]:
    keys = [
        f"{family}:{value}"
        for family, values in identifiers.items()
        for value in values
        if str(value).strip()
    ]
    return tuple(sorted(dict.fromkeys(keys)))


def canonical_entity_key(record: Any) -> str:
    """Return the preferred canonical key for one record."""

    payload = _record_dict(record)
    entity_type = _entity_type(payload)
    identifiers = identifier_keys_for_record(payload)
    if entity_type == "paper" and identifiers.get("doi"):
        return f"paper:doi:{identifiers['doi'][0]}"
    if entity_type == "paper" and identifiers.get("openalex"):
        return f"paper:openalex:{identifiers['openalex'][0]}"
    if entity_type == "paper" and identifiers.get("paper_id"):
        return identifiers["paper_id"][0]
    if entity_type == "dataset" and identifiers.get("dataset_id"):
        return identifiers["dataset_id"][0]
    flattened = _flatten_identifier_keys(identifiers)
    return flattened[0] if flattened else f"unknown:{_clean_token(payload)}"


def _record_primary_id(payload: Mapping[str, Any], index: int) -> str:
    for key in ("dataset_id", "paper_id", "id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    source = str(payload.get("source") or "").strip()
    source_id = str(payload.get("source_id") or "").strip()
    if source and source_id:
        if _entity_type(payload) == "paper":
            return make_paper_id(source, source_id)
        return make_dataset_id(source, source_id)
    return f"record:{index:06d}"


def _preferred_canonical_id(
    entity_type: EntityKind,
    identifiers: Mapping[str, set[str]],
) -> str:
    if entity_type == "paper":
        if identifiers.get("doi"):
            return f"paper:doi:{sorted(identifiers['doi'])[0]}"
        if identifiers.get("openalex"):
            return f"paper:openalex:{sorted(identifiers['openalex'])[0]}"
        if identifiers.get("paper_id"):
            return sorted(identifiers["paper_id"])[0]
    if entity_type == "dataset" and identifiers.get("dataset_id"):
        return sorted(identifiers["dataset_id"])[0]
    flattened = _flatten_identifier_keys(
        {key: sorted(values) for key, values in identifiers.items()}
    )
    return flattened[0] if flattened else "unknown"


def _conflicts_for_group(
    records: list[dict[str, Any]],
    record_ids: tuple[str, ...],
) -> tuple[EntityConflict, ...]:
    conflicts: list[EntityConflict] = []
    for field_name in ("title", "source", "source_id", "doi", "url"):
        values_by_record: dict[str, list[str]] = defaultdict(list)
        for payload, record_id in zip(records, record_ids, strict=True):
            value = str(payload.get(field_name) or "").strip()
            if value:
                values_by_record[value].append(record_id)
        if len(values_by_record) <= 1:
            continue
        conflicts.append(
            EntityConflict(
                field=field_name,
                values=tuple(sorted(values_by_record)),
                record_ids=tuple(
                    sorted(
                        record_id
                        for ids in values_by_record.values()
                        for record_id in ids
                    )
                ),
                resolution="kept_all_values_for_review",
            )
        )
    return tuple(conflicts)


def _component_members(adjacency: Mapping[str, set[str]]) -> list[tuple[str, ...]]:
    seen: set[str] = set()
    components: list[tuple[str, ...]] = []
    for node in sorted(adjacency):
        if node in seen:
            continue
        queue: deque[str] = deque([node])
        seen.add(node)
        members: list[str] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(members)))
    return components


def resolve_entities(records: Iterable[Any]) -> EntityResolutionReport:
    """Resolve records into canonical entities with explicit duplicate groups."""

    payloads = [_record_dict(record) for record in records]
    if not payloads:
        return EntityResolutionReport(entities=(), lookup={}, duplicate_groups=())

    union = _UnionFind()
    record_ids: list[str] = []
    identifier_to_records: dict[str, set[str]] = defaultdict(set)
    records_by_id: dict[str, dict[str, Any]] = {}
    type_by_id: dict[str, EntityKind] = {}

    for index, payload in enumerate(payloads):
        record_id = _record_primary_id(payload, index)
        record_ids.append(record_id)
        records_by_id[record_id] = payload
        type_by_id[record_id] = _entity_type(payload)
        union.add(record_id)
        for identifier in _flatten_identifier_keys(identifier_keys_for_record(payload)):
            identifier_to_records[identifier].add(record_id)

    for ids in identifier_to_records.values():
        ordered = sorted(ids)
        for other in ordered[1:]:
            union.union(ordered[0], other)

    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for record_id in record_ids:
        grouped_ids[union.find(record_id)].append(record_id)

    entities: list[EntityResolutionResult] = []
    lookup: dict[str, str] = {}
    all_conflicts: list[EntityConflict] = []
    duplicate_groups: list[tuple[str, ...]] = []

    for members in sorted(tuple(sorted(ids)) for ids in grouped_ids.values()):
        group_payloads = [records_by_id[record_id] for record_id in members]
        entity_types = sorted({type_by_id[record_id] for record_id in members})
        entity_type: EntityKind = entity_types[0] if len(entity_types) == 1 else "unknown"
        identifiers: dict[str, set[str]] = defaultdict(set)
        aliases: set[str] = set()
        for payload in group_payloads:
            for family, values in identifier_keys_for_record(payload).items():
                identifiers[family].update(values)
            title = str(payload.get("title") or "").strip()
            if title:
                aliases.add(title)
            aliases.update(str(alias) for alias in payload.get("aliases", []) or [] if alias)
        conflicts = _conflicts_for_group(group_payloads, members)
        all_conflicts.extend(conflicts)
        canonical_id = _preferred_canonical_id(entity_type, identifiers)
        for record_id in members:
            lookup[record_id] = canonical_id
        for identifier in _flatten_identifier_keys(identifiers):
            lookup[identifier] = canonical_id
        if len(members) > 1:
            duplicate_groups.append(members)
        entities.append(
            EntityResolutionResult(
                canonical_id=canonical_id,
                entity_type=entity_type,
                record_ids=members,
                aliases=tuple(sorted(aliases)),
                identifiers={
                    family: tuple(sorted(values))
                    for family, values in sorted(identifiers.items())
                },
                conflicts=conflicts,
            )
        )

    entities.sort(key=lambda entity: (entity.entity_type, entity.canonical_id))
    return EntityResolutionReport(
        entities=tuple(entities),
        lookup=dict(sorted(lookup.items())),
        duplicate_groups=tuple(sorted(duplicate_groups)),
        conflicts=tuple(sorted(all_conflicts, key=lambda item: (item.field, item.values))),
    )
