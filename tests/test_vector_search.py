import pytest
import orcachip

def test_engine_properties(engine):
    """Verify engine properties match expected values."""
    assert engine.dimensions == 128
    assert engine.total_vectors > 0
    assert engine.total_bytes > 0
    assert engine.embedding_floats == engine.total_vectors * engine.dimensions


def test_cosine_similarity_search(engine, query):
    """Test Cosine Similarity (Happy path, with and without diagnostics)."""
    # 1. Happy path (diagnostics off)
    res_clean = engine.vector_search(
        query, 
        max_results=5, 
        similarity_metric=orcachip.SimilarityMetric.COSINE, 
        show_diagnostics=False
    )
    assert len(res_clean.matches) <= 5
    assert 0 <= res_clean.matches[0].index < engine.total_vectors

    # 2. With diagnostics
    res_diag = engine.vector_search(
        query, 
        max_results=5, 
        similarity_metric=orcachip.SimilarityMetric.COSINE, 
        show_diagnostics=True
    )
    assert res_diag.diagnostics.search_time >= 0.0
    assert res_diag.diagnostics.total_vectors == engine.total_vectors


def test_other_similarity_metrics(engine, query):
    """Test standard execution for remaining metrics."""
    for metric in [orcachip.SimilarityMetric.DOT_PRODUCT, orcachip.SimilarityMetric.EUCLIDEAN]:
        res = engine.vector_search(query, max_results=3, similarity_metric=metric)
        assert len(res.matches) > 0


def test_error_and_edge_cases(engine, query):
    """Test dimension mismatch error and max_results=0 edge case."""
    # Mismatched query dimensions
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        engine.vector_search([1.0, 2.0])

    # Zero results edge case
    res_zero = engine.vector_search(query, max_results=0)
    assert len(res_zero.matches) == 0