"""Shared dataset/doc-id normalization for whitespace-hostile formats.

A handful of upstream dataset ids (e.g. ``neuromorpho:Physio Lab - Medical
Faculty - UoI``) contain internal whitespace. That is fine in JSON, which quotes
the field, but the TREC qrels/run format is positional and whitespace-delimited
(``query_id 0 doc_id grade``) with no escaping — any embedded space silently
shifts every downstream column and breaks standard tooling such as ``trec_eval``
and ``pytrec_eval``.

``normalize_docid`` collapses internal whitespace runs to a single underscore so
the id is always a single TREC token. It is applied at two boundaries:

* when emitting ``.trec`` files (qrels exporters), and
* when loading run record ids for evaluation,

so both sides of the qrels↔run join are normalized identically and metrics are
unchanged. Canonical JSONL keeps the original, human-readable id.
"""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def normalize_docid(doc_id: str) -> str:
    """Collapse internal whitespace in a doc id to single underscores.

    Idempotent: ``normalize_docid(normalize_docid(x)) == normalize_docid(x)``.
    A doc id with no whitespace is returned unchanged.
    """
    return _WHITESPACE.sub("_", doc_id.strip())
