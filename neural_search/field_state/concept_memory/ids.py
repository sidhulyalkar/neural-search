"""Deterministic stable ID generation for concept memory entities."""

from __future__ import annotations

import hashlib
import re


def _slug(name: str) -> str:
    lowered = name.lower()
    no_punct = re.sub(r"[^\w\s-]", "", lowered)
    return re.sub(r"[\s_]+", "-", no_punct).strip("-")


def concept_id(concept_type: str, name: str) -> str:
    """Generate concept:<type>:<slug>."""
    return f"concept:{concept_type}:{_slug(name)}"


def evidence_id(
    source_concept_id: str,
    target_concept_id: str | None,
    relation_type: str,
) -> str:
    """Generate evidence:<source>:<relation>:<hash6>."""
    raw = f"{source_concept_id}{target_concept_id or ''}{relation_type}"
    hash6 = hashlib.sha256(raw.encode()).hexdigest()[:6]
    return f"evidence:{source_concept_id}:{relation_type}:{hash6}"


def basis_id(concept_id_str: str) -> str:
    """Generate basis:<concept_id>."""
    return f"basis:{concept_id_str}"
