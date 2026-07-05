"""DuckDB-backed knowledge graph store.

Materialises nodes and edges from composed_kg.jsonl into indexed DuckDB tables,
providing fast SQL queries without loading the full 100 MB JSONL into memory.

Tables
------
kg_nodes  — one row per node (node_id, node_type, label, properties JSON)
kg_edges  — one row per edge (edge_id, source_id, target_id, edge_type, weight, properties JSON)

Typical usage
-------------
    conn = load_kg_store()                         # cached connection
    nodes = query_nodes(conn, node_type="dataset", limit=50)
    edges = query_edges(conn, source_id="node:dataset:dandi:000001")
    node  = get_node(conn, "node:dataset:dandi:000001")

Build the store from scratch
-----------------------------
    conn = duckdb.connect("data/kg/neural_search_kg.duckdb")
    init_kg_tables(conn)
    import_composed_kg(conn, Path("artifacts/kg/composed_kg.jsonl"))
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/kg/neural_search_kg.duckdb")
DEFAULT_JSONL_PATH = Path("artifacts/kg/composed_kg.jsonl")

# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_KG_NODES = """
CREATE TABLE IF NOT EXISTS kg_nodes (
    node_id    TEXT PRIMARY KEY,
    node_type  TEXT,
    label      TEXT,
    properties JSON
)
"""

_CREATE_KG_EDGES = """
CREATE TABLE IF NOT EXISTS kg_edges (
    edge_id    TEXT,
    source_id  TEXT,
    target_id  TEXT,
    edge_type  TEXT,
    weight     REAL,
    properties JSON
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_kg_nodes_type ON kg_nodes (node_type)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges (source_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges (target_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_type   ON kg_edges (edge_type)",
]


def init_kg_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create kg_nodes and kg_edges tables and their indexes if they don't exist."""
    conn.execute(_CREATE_KG_NODES)
    conn.execute(_CREATE_KG_EDGES)
    for idx_sql in _INDEXES:
        conn.execute(idx_sql)


# ── Record classification ─────────────────────────────────────────────────────

def _is_node(record: dict[str, Any]) -> bool:
    """Return True when the JSONL record represents a KG node."""
    return "node_id" in record or "node_type" in record


def _is_edge(record: dict[str, Any]) -> bool:
    """Return True when the JSONL record represents a KG edge."""
    return (
        "source_id" in record
        or "source_node_id" in record
        or "edge_type" in record
    ) and "node_type" not in record


def _extract_node_row(record: dict[str, Any]) -> tuple[str, str, str, str] | None:
    """Extract (node_id, node_type, label, properties_json) from a record, or None."""
    node_id = record.get("node_id") or record.get("id")
    node_type = record.get("node_type") or record.get("type") or ""
    label = record.get("label") or record.get("name") or node_id or ""
    if not node_id:
        return None
    props = {k: v for k, v in record.items() if k not in {"node_id", "id", "node_type", "type", "label", "name"}}
    return (str(node_id), str(node_type), str(label), json.dumps(props))


def _extract_edge_row(record: dict[str, Any]) -> tuple[str, str, str, str, float, str] | None:
    """Extract (edge_id, source_id, target_id, edge_type, weight, properties_json), or None."""
    source_id = record.get("source_id") or record.get("source_node_id") or record.get("source")
    target_id = record.get("target_id") or record.get("target_node_id") or record.get("target")
    edge_type = record.get("edge_type") or record.get("relation") or record.get("type") or ""
    if not source_id or not target_id:
        return None
    edge_id = (
        record.get("edge_id")
        or record.get("id")
        or f"edge:{source_id}:{edge_type}:{target_id}"
    )
    weight = float(record.get("weight") or record.get("confidence") or 1.0)
    props = {
        k: v
        for k, v in record.items()
        if k not in {
            "edge_id", "id", "source_id", "source_node_id", "source",
            "target_id", "target_node_id", "target", "edge_type", "relation",
            "type", "weight", "confidence",
        }
    }
    return (str(edge_id), str(source_id), str(target_id), str(edge_type), weight, json.dumps(props))


# ── Import ─────────────────────────────────────────────────────────────────────

_BATCH_SIZE = 5_000


def import_composed_kg(
    conn: duckdb.DuckDBPyConnection,
    jsonl_path: Path | str,
) -> dict[str, int]:
    """Bulk-import nodes and edges from composed_kg.jsonl into DuckDB.

    Idempotent: uses INSERT OR IGNORE for nodes (PRIMARY KEY) and skips
    duplicate edges by checking existence before inserting.

    Returns a dict with ``nodes_inserted`` and ``edges_inserted`` counts.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        log.warning("KG JSONL not found at %s; skipping import", jsonl_path)
        return {"nodes_inserted": 0, "edges_inserted": 0}

    node_rows: list[tuple] = []
    edge_rows: list[tuple] = []
    nodes_total = edges_total = 0
    skipped = 0

    def _flush_nodes() -> None:
        nonlocal nodes_total
        if not node_rows:
            return
        conn.executemany(
            "INSERT OR IGNORE INTO kg_nodes (node_id, node_type, label, properties) "
            "VALUES (?, ?, ?, ?)",
            node_rows,
        )
        nodes_total += len(node_rows)
        node_rows.clear()

    def _flush_edges() -> None:
        nonlocal edges_total
        if not edge_rows:
            return
        conn.executemany(
            "INSERT INTO kg_edges (edge_id, source_id, target_id, edge_type, weight, properties) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            edge_rows,
        )
        edges_total += len(edge_rows)
        edge_rows.clear()

    with jsonl_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record: dict[str, Any] = json.loads(raw_line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Unwrap envelope format: {"record_type": "node"|"edge", "node"|"edge": {...}}
            if "record_type" in record:
                rt = record["record_type"]
                inner = record.get(rt)
                if not isinstance(inner, dict):
                    skipped += 1
                    continue
                record = inner

            if _is_node(record):
                row = _extract_node_row(record)
                if row:
                    node_rows.append(row)
                    if len(node_rows) >= _BATCH_SIZE:
                        _flush_nodes()
            elif _is_edge(record):
                row = _extract_edge_row(record)
                if row:
                    edge_rows.append(row)
                    if len(edge_rows) >= _BATCH_SIZE:
                        _flush_edges()
            else:
                skipped += 1

    _flush_nodes()
    _flush_edges()

    if skipped:
        log.debug("import_composed_kg: skipped %d unclassifiable records", skipped)
    log.info("import_composed_kg: inserted %d nodes, %d edges", nodes_total, edges_total)
    return {"nodes_inserted": nodes_total, "edges_inserted": edges_total}


# ── Query helpers ──────────────────────────────────────────────────────────────

def query_nodes(
    conn: duckdb.DuckDBPyConnection,
    *,
    node_type: str | None = None,
    label_contains: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return nodes matching optional filters as a list of dicts."""
    clauses: list[str] = []
    params: list[Any] = []

    if node_type is not None:
        clauses.append("node_type = ?")
        params.append(node_type)
    if label_contains is not None:
        clauses.append("lower(label) LIKE ?")
        params.append(f"%{label_contains.lower()}%")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT node_id, node_type, label, properties FROM kg_nodes {where} LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "node_id": r[0],
            "node_type": r[1],
            "label": r[2],
            "properties": json.loads(r[3]) if r[3] else {},
        }
        for r in rows
    ]


def query_edges(
    conn: duckdb.DuckDBPyConnection,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
    edge_type: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return edges matching optional filters as a list of dicts."""
    clauses: list[str] = []
    params: list[Any] = []

    if source_id is not None:
        clauses.append("source_id = ?")
        params.append(source_id)
    if target_id is not None:
        clauses.append("target_id = ?")
        params.append(target_id)
    if edge_type is not None:
        clauses.append("edge_type = ?")
        params.append(edge_type)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        f"SELECT edge_id, source_id, target_id, edge_type, weight, properties "
        f"FROM kg_edges {where} LIMIT ?"
    )
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "edge_id": r[0],
            "source_id": r[1],
            "target_id": r[2],
            "edge_type": r[3],
            "weight": float(r[4]) if r[4] is not None else 1.0,
            "properties": json.loads(r[5]) if r[5] else {},
        }
        for r in rows
    ]


def get_node(
    conn: duckdb.DuckDBPyConnection,
    node_id: str,
) -> dict[str, Any] | None:
    """Fetch a single node by ID, returning a dict or None when absent."""
    row = conn.execute(
        "SELECT node_id, node_type, label, properties FROM kg_nodes WHERE node_id = ?",
        [node_id],
    ).fetchone()
    if row is None:
        return None
    return {
        "node_id": row[0],
        "node_type": row[1],
        "label": row[2],
        "properties": json.loads(row[3]) if row[3] else {},
    }


# ── Cached connection ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_kg_store(
    db_path: str = str(DEFAULT_DB_PATH),
    jsonl_path: str = str(DEFAULT_JSONL_PATH),
) -> duckdb.DuckDBPyConnection:
    """Open (or create) the KG DuckDB store, initialise tables, and import if empty.

    LRU-cached so the connection is reused across requests without re-reading
    the JSONL on every call.
    """
    resolved_db = Path(db_path)
    resolved_db.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(resolved_db))
    init_kg_tables(conn)

    node_count: int = conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
    if node_count == 0:
        log.info("kg_nodes is empty — importing from %s", jsonl_path)
        import_composed_kg(conn, Path(jsonl_path))
    else:
        log.debug("load_kg_store: %d nodes already in %s", node_count, db_path)

    return conn
