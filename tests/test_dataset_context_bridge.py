"""Tests for dataset_context_bridge module."""
from neural_search.retrieval.dataset_context_bridge import dataset_context_from_record
from neural_search.retrieval.usefulness_scorer import DatasetContext


def _make_record(
    dataset_id="test_ds",
    tasks=None,
    modalities=None,
    species=None,
    regions=None,
):
    return {
        "id": dataset_id,
        "tasks": [{"label": t} for t in (tasks or [])],
        "modalities": [{"label": m} for m in (modalities or [])],
        "species": [{"label": s} for s in (species or [])],
        "brain_regions": [{"label": r} for r in (regions or [])],
    }


def test_returns_dataset_context():
    record = _make_record("ds001", tasks=["decision-making"])
    ctx = dataset_context_from_record(record)
    assert isinstance(ctx, DatasetContext)


def test_dataset_id_extracted():
    record = _make_record("ds_abc")
    ctx = dataset_context_from_record(record)
    assert ctx.dataset_id == "ds_abc"


def test_tasks_extracted_from_label_dicts():
    record = _make_record(tasks=["reversal_learning", "go_nogo"])
    ctx = dataset_context_from_record(record)
    assert set(ctx.tasks) == {"reversal_learning", "go_nogo"}


def test_modalities_extracted():
    record = _make_record(modalities=["neuropixels", "calcium_imaging"])
    ctx = dataset_context_from_record(record)
    assert set(ctx.modalities) == {"neuropixels", "calcium_imaging"}


def test_empty_record_returns_default_context():
    ctx = dataset_context_from_record({})
    assert ctx.dataset_id == ""
    assert ctx.tasks == []
    assert ctx.modalities == []


def test_card_affordances_extracted():
    record = _make_record("ds_aff")
    card = {"analysis_affordances": [{"affordance_id": "choice_decoding"}, {"affordance_id": "glm_ready"}]}
    ctx = dataset_context_from_record(record, card)
    assert set(ctx.affordances) == {"choice_decoding", "glm_ready"}


def test_card_data_standards_extracted():
    record = _make_record("ds_std")
    card = {"data_standards": ["NWB", "BIDS"]}
    ctx = dataset_context_from_record(record, card)
    assert set(ctx.data_standards) == {"NWB", "BIDS"}


def test_session_count_from_record():
    record = _make_record("ds_sess")
    record["session_count"] = 42
    ctx = dataset_context_from_record(record)
    assert ctx.session_count == 42


def test_quality_score_from_card():
    record = _make_record("ds_q")
    card = {"quality_score": 0.87}
    ctx = dataset_context_from_record(record, card)
    assert abs(ctx.quality_score - 0.87) < 1e-6


def test_card_n_sessions_key_used():
    """DatasetCardV1 uses n_sessions, not session_count."""
    record = _make_record("ds_n")
    card = {"n_sessions": 10, "n_trials": 200, "n_subjects": 5}
    ctx = dataset_context_from_record(record, card)
    assert ctx.session_count == 10
    assert ctx.trial_count == 200
    assert ctx.subject_count == 5


def test_quality_score_from_dict_card():
    """DatasetCardV1 quality_score serializes as a dict with overall_score."""
    record = _make_record("ds_qs")
    card = {"quality_score": {"overall_score": 0.85, "completeness": 0.9}}
    ctx = dataset_context_from_record(record, card)
    assert abs(ctx.quality_score - 0.85) < 1e-6


def test_data_standards_no_empty_strings():
    """Dict-format data_standards with missing name/id must not produce empty strings."""
    record = _make_record("ds_std2")
    card = {"data_standards": [{"foo": "bar"}, "NWB"]}
    ctx = dataset_context_from_record(record, card)
    assert "" not in ctx.data_standards
    assert "NWB" in ctx.data_standards
