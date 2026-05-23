from neural_search.latent import extract_session_features
from neural_search.latent.embedding_schema import FeatureType
from neural_search.latent.search import (
    search_by_behavior_pattern,
    search_by_neural_similarity,
)


def _dataset(source_id: str, **overrides):
    dataset = {
        "id": source_id,
        "source_id": source_id,
        "modalities": ["extracellular_ephys"],
        "tasks": ["reversal_learning"],
        "behaviors": ["cue", "choice", "reward", "omission"],
        "has_trials": True,
        "has_behavior": True,
        "license": "CC-BY-4.0",
        "data_standards": ["NWB"],
    }
    dataset.update(overrides)
    return dataset


def test_extract_session_features_is_deterministic():
    dataset = _dataset("REVERSAL")
    session_data = {
        "session_id": "s1",
        "events": ["cue", "choice", "reward"],
        "neural_stats": {"unit_count": 200, "mean_firing_rate_hz": 5},
        "qc": {"artifact_ratio": 0.05, "metadata_completeness": 0.9},
    }

    first = extract_session_features(dataset, session_data)
    second = extract_session_features(dataset, session_data)

    assert first.to_dict()["features"] == second.to_dict()["features"]
    assert first.has_neural_features
    assert first.has_behavior_features
    assert {feature.feature_type for feature in first.features} >= {
        FeatureType.EVENT_HISTOGRAM,
        FeatureType.NEURAL_SUMMARY_STATISTICS,
        FeatureType.SESSION_QC_VECTOR,
    }


def test_latent_similarity_prefers_matching_sessions():
    query = extract_session_features(_dataset("QUERY"))
    matching = extract_session_features(_dataset("MATCH"))
    mismatch = extract_session_features(
        _dataset(
            "MISMATCH",
            modalities=["fmri"],
            tasks=["visual_decision_making"],
            behaviors=["stimulus", "response", "correct"],
        )
    )

    neural_results = search_by_neural_similarity(query, [matching, mismatch], top_k=2)
    behavior_results = search_by_behavior_pattern(query, [matching, mismatch], top_k=2)

    assert neural_results[0].dataset_id == "MATCH"
    assert behavior_results[0].dataset_id == "MATCH"
