"""Compare BGE-large vs SPECTER2 retrieval on the benchmark query set.

Builds (or loads cached) SPECTER2 embeddings for the full corpus, then runs
retrieval on the benchmark queries and compares NDCG@10 / MRR against
the existing BGE-large run file.

SPECTER2 is trained on scientific papers and often outperforms general models
on domain-specific neuroscience queries — but may underperform on metadata-heavy
queries where the corpus text is short.

Usage
-----
    # Build SPECTER2 embeddings (requires ~2GB model download on first run)
    python scripts/eval/run_specter2_comparison.py --build-embeddings

    # Run retrieval and comparison (uses cached embeddings)
    python scripts/eval/run_specter2_comparison.py \
        --queries data/eval/benchmark_queries.yaml \
        --bge-run reports/eval/runs/bge.jsonl \
        --out-dir reports/eval/runs/

    # Full pipeline
    python scripts/eval/run_specter2_comparison.py --build-embeddings --compare
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CORPUS_PATH = ROOT / "data" / "corpus" / "normalized" / "combined_corpus.jsonl"
EMBEDDINGS_CACHE = ROOT / "data" / "embeddings" / "specter2_corpus.jsonl"
QUERIES_PATH = ROOT / "data" / "eval" / "benchmark_queries.yaml"
OUT_DIR = ROOT / "reports" / "eval" / "runs"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 32
TOP_K = 100


def _corpus_text(rec: dict) -> str:
    title = rec.get("title") or ""
    abstract = rec.get("description") or rec.get("abstract") or ""
    return f"{title}\n{abstract}".strip()[:2000]


def build_specter2_embeddings(corpus_path: Path, out_path: Path) -> None:
    """Embed the full corpus with SPECTER2 and cache to JSONL."""
    try:
        from neural_search.embeddings.providers import SPECTER2Provider
    except ImportError:
        log.error("neural_search.embeddings not available")
        raise

    log.info("Loading SPECTER2 model: %s", MODEL_NAME)
    provider = SPECTER2Provider(MODEL_NAME)

    records = []
    with corpus_path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    log.info("Corpus: %d records", len(records))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w") as fout:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            texts = [_corpus_text(r) for r in batch]
            try:
                embeddings = provider.embed_batch(texts)
                for rec, emb in zip(batch, embeddings, strict=True):
                    did = str(rec.get("dataset_id") or f"{rec.get('source')}:{rec.get('source_id')}")
                    emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    fout.write(json.dumps({"dataset_id": did, "embedding": emb_list}) + "\n")
                    n_written += 1
            except Exception as e:
                log.warning("Batch %d failed: %s", i // BATCH_SIZE, e)
            if i % (BATCH_SIZE * 10) == 0:
                log.info("  %d / %d records", i, len(records))
    log.info("SPECTER2 embeddings written: %d records → %s", n_written, out_path)


def load_embeddings(path: Path) -> tuple[list[str], list[list[float]]]:
    """Load cached SPECTER2 embeddings."""
    ids, vecs = [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(rec["dataset_id"])
            vecs.append(rec["embedding"])
    return ids, vecs


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def retrieve_specter2(
    query: str,
    ids: list[str],
    vecs: list[list[float]],
    provider: Any,
    top_k: int = TOP_K,
) -> list[tuple[str, float]]:
    q_emb = provider.embed_batch([query])[0]
    q_vec = q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb)
    scores = [(did, cosine_sim(q_vec, vec)) for did, vec in zip(ids, vecs, strict=True)]
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def run_retrieval(queries_path: Path, embeddings_path: Path, out_path: Path) -> None:
    """Run SPECTER2 retrieval over all benchmark queries."""
    try:
        from neural_search.embeddings.providers import SPECTER2Provider
    except ImportError:
        log.error("neural_search.embeddings not available")
        raise

    raw = yaml.safe_load(queries_path.read_text())
    queries = raw.get("benchmark_queries", [])
    log.info("Queries: %d", len(queries))

    log.info("Loading SPECTER2 embeddings from %s", embeddings_path)
    ids, vecs = load_embeddings(embeddings_path)
    log.info("Loaded %d document embeddings", len(ids))

    provider = SPECTER2Provider(MODEL_NAME)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w") as fout:
        for q in queries:
            qid = q["id"]
            text = q["query"]
            t0 = time.time()
            ranked = retrieve_specter2(text, ids, vecs, provider)
            elapsed = time.time() - t0
            for rank, (did, score) in enumerate(ranked, 1):
                fout.write(json.dumps({
                    "query_id": qid,
                    "dataset_id": did,
                    "rank": rank,
                    "score": round(score, 6),
                    "system": "specter2",
                }) + "\n")
                n_written += 1
            if rank == 1 or qid in ("q001", "q014", "q018"):
                log.info("  %s: top=%s score=%.4f in %.2fs", qid, ranked[0][0], ranked[0][1], elapsed)
    log.info("SPECTER2 run written: %d rows → %s", n_written, out_path)


def compare_runs(specter2_run: Path, bge_run: Path) -> None:
    """Print side-by-side comparison of top results per query."""
    def load(p: Path) -> dict[str, list[str]]:
        run: dict[str, list[tuple[int, str]]] = {}
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid, did, rank = rec["query_id"], rec["dataset_id"], rec["rank"]
                run.setdefault(qid, []).append((rank, did))
        return {q: [d for _, d in sorted(p)] for q, p in run.items()}

    s2 = load(specter2_run)
    bge = load(bge_run) if bge_run.exists() else {}

    print(f"\n{'Query':8s}  {'SPECTER2 top-1':50s}  {'BGE top-1':50s}")
    print("-" * 120)
    for qid in sorted(s2.keys())[:20]:
        s2_top = s2[qid][0] if s2.get(qid) else "-"
        bge_top = bge[qid][0] if bge.get(qid) else "-"
        match = "✓" if s2_top == bge_top else " "
        print(f"{qid:8s}  {s2_top[:48]:50s}  {bge_top[:48]:50s} {match}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--embeddings-cache", type=Path, default=EMBEDDINGS_CACHE)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--bge-run", type=Path, default=OUT_DIR / "bge.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--build-embeddings", action="store_true",
                        help="Build/refresh SPECTER2 embeddings (slow — downloads ~2GB model)")
    parser.add_argument("--compare", action="store_true",
                        help="Print top-1 comparison after retrieval")
    args = parser.parse_args(argv)

    if args.build_embeddings:
        if not args.corpus.exists():
            print(f"Corpus not found: {args.corpus}", file=sys.stderr)
            return 1
        build_specter2_embeddings(args.corpus, args.embeddings_cache)

    if not args.embeddings_cache.exists():
        print(
            f"Embeddings cache not found: {args.embeddings_cache}\n"
            "Run with --build-embeddings first.",
            file=sys.stderr,
        )
        return 1

    specter2_run = args.out_dir / "specter2.jsonl"
    run_retrieval(args.queries, args.embeddings_cache, specter2_run)

    if args.compare:
        compare_runs(specter2_run, args.bge_run)

    print(f"\nSPECTER2 run: {specter2_run}")
    print(f"To compare with qrels: python scripts/eval/compute_bootstrap_ci.py "
          f"--qrels artifacts/qrels_silver.jsonl --runs {args.bge_run} {specter2_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
