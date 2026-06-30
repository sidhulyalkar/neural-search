"""Build the citation graph KG layer from 9M citation edges.

Streams `artifacts/citations/citation_edges.jsonl` and emits
`paper_cites_paper` edges only for papers present in our corpus.

Strategy:
  1. Build an index of known paper IDs from openalex_neuro batches.
  2. Stream the citation JSONL (never fully loaded — 9M rows).
  3. Emit KG edges for any (citing, cited) pair where BOTH IDs are known.

Edge type: paper_cites_paper

Output statistics vary by corpus size:
  ~26 OpenAlex batches → likely 50K–300K in-corpus citation edges.

Run: python -m neural_search.ingestion.citation_builder
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from neural_search.graph.schema import GraphNode, GraphEdge, KnowledgeGraph

log = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
CITATION_JSONL = ARTIFACTS_DIR / "citations" / "citation_edges.jsonl"

OPENALEX_DIR = DATA_ROOT / "corpus" / "normalized" / "openalex_neuro"
CORPUS_JSONL_DIR = DATA_ROOT / "corpus" / "normalized" / "combined_corpus.jsonl"

# Cap to avoid enormous KG objects; increase for batch pipeline use
MAX_EDGES = 500_000


def _build_known_paper_ids() -> set[str]:
    """Return set of bare OpenAlex IDs (e.g. 'W2963345511') from our corpus."""
    known: set[str] = set()

    if OPENALEX_DIR.exists():
        for fp in sorted(OPENALEX_DIR.glob("*.jsonl")):
            try:
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    sid = obj.get("source_id") or obj.get("paper_id", "")
                    # Normalise to bare W-ID
                    bare = sid.removeprefix("paper:openalex:")
                    if bare.startswith("W"):
                        known.add(bare)
            except Exception as exc:
                log.debug("Skipping %s: %s", fp.name, exc)

    if CORPUS_JSONL_DIR.exists():
        for fp in CORPUS_JSONL_DIR.glob("*.papers.jsonl"):
            try:
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    sid = obj.get("source_id") or obj.get("paper_id", "")
                    bare = sid.removeprefix("paper:openalex:")
                    if bare.startswith("W"):
                        known.add(bare)
            except Exception:
                pass

    log.info("Citation builder: %d known corpus paper IDs.", len(known))
    return known


def build_citation_nodes() -> list[GraphNode]:
    # Paper nodes already created by the corpus ingestion layer; return empty.
    return []


def build_citation_edges(max_edges: int = MAX_EDGES) -> list[GraphEdge]:
    if not CITATION_JSONL.exists():
        log.warning(
            "Citation edges file not found at %s. "
            "Run the citation ingestion pipeline first.",
            CITATION_JSONL,
        )
        return []

    known_ids = _build_known_paper_ids()
    if not known_ids:
        log.warning("No corpus paper IDs found — citation graph will be empty.")
        return []

    edges: list[GraphEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    total_read = 0
    in_corpus = 0
    skipped_unknown = 0

    log.info("Streaming citation edges from %s…", CITATION_JSONL)

    with open(CITATION_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_read += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            citing_raw = obj.get("citing_paper_id", "")
            cited_raw = obj.get("cited_paper_id", "")
            year = obj.get("citing_year")

            # Normalise to bare W-IDs
            citing_bare = citing_raw.removeprefix("paper:openalex:") if citing_raw else ""
            cited_bare = cited_raw.removeprefix("paper:openalex:") if cited_raw else ""

            if not citing_bare or not cited_bare:
                continue

            # Only keep in-corpus pairs
            if citing_bare not in known_ids or cited_bare not in known_ids:
                skipped_unknown += 1
                continue

            pair = (citing_bare, cited_bare)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            citing_node_id = f"paper:openalex:{citing_bare}"
            cited_node_id = f"paper:openalex:{cited_bare}"
            edge_id = f"edge:cite:{citing_bare}:{cited_bare}"

            edges.append(
                GraphEdge(
                    edge_id=edge_id,
                    source_node_id=citing_node_id,
                    target_node_id=cited_node_id,
                    edge_type="paper_cites_paper",
                    confidence=1.0,
                    properties={
                        "citing_year": year,
                        "source": "citation_graph",
                    },
                )
            )
            in_corpus += 1

            if in_corpus % 50_000 == 0:
                log.info(
                    "  %d in-corpus citation edges so far (read %d total)…",
                    in_corpus, total_read,
                )

            if in_corpus >= max_edges:
                log.info("Reached max_edges cap (%d). Stopping.", max_edges)
                break

    log.info(
        "Citation builder: %d in-corpus edges from %d total rows (%d skipped as out-of-corpus).",
        in_corpus, total_read, skipped_unknown,
    )
    return edges


def build_citation_kg(max_edges: int = MAX_EDGES) -> KnowledgeGraph:
    nodes = build_citation_nodes()
    edges = build_citation_edges(max_edges=max_edges)
    log.info("Citation KG: %d nodes, %d paper_cites_paper edges", len(nodes), len(edges))
    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-edges", type=int, default=MAX_EDGES,
        help=f"Max citation edges to emit (default: {MAX_EDGES})"
    )
    args = parser.parse_args()

    kg = build_citation_kg(max_edges=args.max_edges)
    print(f"Citation KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    if kg.edges:
        sample = list(kg.edges.values())[:5]
        for e in sample:
            citing = e.source_node_id  # e.g. paper:openalex:W123
            cited = e.target_node_id
            yr = e.properties.get("citing_year", "?")
            print(f"  {citing} ({yr}) -> {cited}")
