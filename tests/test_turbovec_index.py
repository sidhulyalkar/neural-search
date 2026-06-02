"""Tests for NeuralSearchTurboIndex."""
import numpy as np
import pytest


def test_turbovec_import():
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    assert NeuralSearchTurboIndex is not None


def test_index_add_and_search():
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex(dim=4, bit_width=4)
    vecs = np.random.randn(10, 4).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"ds{i:03d}" for i in range(10)]
    idx.add(ids=ids, vectors=vecs)
    assert idx.size == 10

    q = vecs[0].copy()
    results = idx.search(q, k=3)
    assert len(results) == 3
    assert results[0][0] == ids[0]


def test_index_save_load(tmp_path):
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex(dim=4, bit_width=4)
    vecs = np.eye(4, dtype=np.float32)
    ids = ["a", "b", "c", "d"]
    idx.add(ids=ids, vectors=vecs)

    path = str(tmp_path / "test.turbo")
    idx.save(path)

    idx2 = NeuralSearchTurboIndex.load(path)
    assert idx2.size == 4
    results = idx2.search(vecs[0], k=2)
    assert results[0][0] == "a"


def test_index_provider_metadata():
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex(dim=1024, bit_width=4)
    assert idx.dim == 1024
    assert idx.bit_width == 4


def test_invalid_bit_width():
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    with pytest.raises(ValueError, match="bit_width"):
        NeuralSearchTurboIndex(dim=4, bit_width=8)


def test_search_empty_index():
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex(dim=4, bit_width=4)
    q = np.ones(4, dtype=np.float32)
    results = idx.search(q, k=5)
    assert results == []
