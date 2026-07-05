"""Wrapper for turbovec IdMapIndex with stable save/load API.

IdMapIndex preserves stable string IDs (unlike TurboQuantIndex).
bit_width must be 2 or 4 — 8 is not supported.
Falls back to exact brute-force cosine search if turbovec not installed.

Usage:
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    import numpy as np

    idx = NeuralSearchTurboIndex(dim=1024, bit_width=4)
    idx.add(ids=["dataset:dandi:000003", ...], vectors=embeddings)  # (N, 1024) float32

    results = idx.search(query_vec, k=50)
    # returns list of (dataset_id: str, distance: float), distance ascending

    idx.save("data/index/turbovec.index")
    idx2 = NeuralSearchTurboIndex.load("data/index/turbovec.index")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TURBOVEC_AVAILABLE: bool | None = None


def _check_turbovec() -> bool:
    global _TURBOVEC_AVAILABLE
    if _TURBOVEC_AVAILABLE is None:
        try:
            import turbovec  # noqa: F401
            _TURBOVEC_AVAILABLE = True
        except ImportError:
            _TURBOVEC_AVAILABLE = False
    return _TURBOVEC_AVAILABLE


class NeuralSearchTurboIndex:
    """Compressed ANN index backed by turbovec IdMapIndex.

    Falls back to exact brute-force cosine search if turbovec is not installed.
    bit_width must be 2 or 4. At 5000 datasets, 4-bit uses ~2.6 MB vs ~20 MB float32.
    """

    def __init__(self, dim: int = 1024, bit_width: int = 4) -> None:
        if bit_width not in (2, 4):
            raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")
        self.dim = dim
        self.bit_width = bit_width
        self._ids: list[str] = []
        self._vecs: np.ndarray | None = None

        if _check_turbovec():
            from turbovec import IdMapIndex
            self._index: Any = IdMapIndex(dim=dim, bit_width=bit_width)
        else:
            logger.warning(
                "turbovec not installed — using exact brute-force fallback. "
                "Install turbovec for production use: pip install turbovec"
            )
            self._index = None

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Add vectors with stable string IDs.

        Args:
            ids: String dataset IDs.
            vectors: Float32 array of shape (len(ids), dim). Should be L2-normalized.
        """
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"ids length {len(ids)} != vectors rows {vectors.shape[0]}")
        vecs = vectors.astype(np.float32)

        if self._index is not None:
            # The installed turbovec API (IdMapIndex.add_with_ids) only accepts
            # uint64 external ids, not our string dataset ids -- use each
            # vector's position in self._ids as its uint64 id and map back to
            # the string id in search().
            start = len(self._ids)
            uint_ids = np.arange(start, start + len(ids), dtype=np.uint64)
            self._index.add_with_ids(vectors=vecs, ids=uint_ids)
        else:
            self._vecs = vecs if self._vecs is None else np.vstack([self._vecs, vecs])

        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        """Search for k nearest neighbors.

        Returns list of (dataset_id, distance) sorted by distance ascending.
        """
        if not self._ids:
            return []
        k = min(k, len(self._ids))
        q = query.astype(np.float32)

        if self._index is not None:
            # IdMapIndex.search() requires a 2-D (n_queries, dim) batch and
            # returns (scores, ids) as (n_queries, k) arrays, uint64 ids, with
            # *descending* similarity scores -- not the ascending-distance
            # single-query contract this wrapper promises. Reshape/unpack the
            # single-query batch and convert similarity -> distance the same
            # way the brute-force fallback below does (distance = 1 - sim) so
            # both code paths return the same ordering and units.
            scores, ids = self._index.search(q.reshape(1, -1), k)
            scores, ids = scores[0], ids[0]
            return [
                (self._ids[int(i)], float(1.0 - s))
                for i, s in zip(ids, scores, strict=True)
            ]

        if self._vecs is None:
            return []
        sims = self._vecs @ q
        top_idx = np.argsort(sims)[::-1][:k]
        return [(self._ids[i], float(1.0 - sims[i])) for i in top_idx]

    def save(self, path: str) -> None:
        """Save index to disk. Creates a directory with index + metadata."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        meta = {"dim": self.dim, "bit_width": self.bit_width, "ids": self._ids}
        (p / "meta.json").write_text(json.dumps(meta))

        if self._index is not None:
            self._index.write(str(p / "turbovec.bin"))
        elif self._vecs is not None:
            np.save(str(p / "fallback_vecs.npy"), self._vecs)

        logger.info("Saved %d-record index → %s", self.size, p)

    @classmethod
    def load(cls, path: str) -> NeuralSearchTurboIndex:
        """Load index from disk."""
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        obj = cls(dim=meta["dim"], bit_width=meta["bit_width"])
        obj._ids = meta["ids"]

        turbo_bin = p / "turbovec.bin"
        fallback_npy = p / "fallback_vecs.npy"

        if turbo_bin.exists() and obj._index is not None:
            # IdMapIndex.load() is a classmethod that returns a new instance
            # (there is no in-place instance .load()) -- replace the empty
            # index __init__ constructed rather than trying to mutate it.
            from turbovec import IdMapIndex
            obj._index = IdMapIndex.load(str(turbo_bin))
        elif fallback_npy.exists():
            obj._vecs = np.load(str(fallback_npy))
        elif turbo_bin.exists() and obj._index is None:
            # Index was saved with turbovec but turbovec is not installed here.
            # _ids is populated but search() will return [] until turbovec is installed.
            logger.warning(
                "Index at %s was saved with turbovec but turbovec is not installed. "
                "search() will return empty results. Install turbovec: pip install turbovec",
                p,
            )

        logger.info("Loaded %d-record index ← %s", obj.size, p)
        return obj
