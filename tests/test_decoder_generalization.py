"""Protocol, provenance, split leakage, and deterministic pilot tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neural_search.decoder_generalization.__main__ import main
from neural_search.decoder_generalization.sources import (
    load_bnci2014_001,
    load_npz,
    repair_known_synthetic_channel_map,
    synthetic_dataset,
)
from neural_search.decoder_generalization.study import (
    TrialDataset,
    build_folds,
    run_study,
)


def test_group_splits_enforce_subject_and_session_boundaries():
    d = synthetic_dataset()
    folds = build_folds(d)
    assert sum(f.protocol == "cross_subject" for f in folds) == 4
    assert sum(f.protocol == "cross_session" for f in folds) == 8
    assert sum(f.protocol == "within_session" for f in folds) == 24
    assert sum(f.protocol == "within_session_runout" for f in folds) == 32
    for f in folds:
        assert not set(f.train) & set(f.test)
        if f.protocol == "cross_session":
            assert set(d.subject[f.train]) == set(d.subject[f.test])
            assert not set(d.session[f.train]) & set(d.session[f.test])
        if f.protocol == "within_session_runout":
            assert set(d.subject[f.train]) == set(d.subject[f.test])
            assert set(d.session[f.train]) == set(d.session[f.test])
            assert not set(d.run[f.train]) & set(d.run[f.test])
        if f.protocol == "cross_subject":
            assert not set(d.subject[f.train]) & set(d.subject[f.test])
        if f.protocol == "within_session":
            assert set(d.subject[f.train]) == set(d.subject[f.test])
            assert set(d.session[f.train]) == set(d.session[f.test])


def test_determinism_and_diagnostics():
    d = synthetic_dataset(seed=31)
    r1 = run_study(d, seed=31)
    r2 = run_study(d, seed=31)
    for report in (r1, r2):
        assert report["summary"]["within_session"]["folds"] == 24
        assert report["summary"]["within_session_runout"]["folds"] == 32
        assert report["dataset"]["has_run_metadata"]
        assert report["summary"]["cross_session"]["folds"] == 8
        assert report["summary"]["cross_subject"]["folds"] == 4
        assert len(report["paired_within_to_cross_session_gaps"]) == 8
        assert {
            r["reference_protocol"]
            for r in report["paired_within_to_cross_session_gaps"]
        } == {"within_session_runout"}
        assert report["subject_clustered_uncertainty"]["n_subjects"] == 4
        assert report["subject_clustered_uncertainty"]["ci95"] is not None
        assert all(0 <= row["balanced_accuracy"] <= 1 for row in report["results"] if row["status"] == "evaluated")
        assert all("feature_shift_z" in row for row in report["results"] if row["status"] == "evaluated")
    assert r1["results"] == r2["results"]
    assert (
        r1["subject_clustered_uncertainty"]
        == r2["subject_clustered_uncertainty"]
    )
    assert r1["dataset"]["fingerprint_sha256"] == r2["dataset"]["fingerprint_sha256"]


def test_identical_epoch_in_different_sessions_is_hard_failure():
    d = synthetic_dataset()
    x = d.X.copy()
    a = np.flatnonzero((d.subject == "s1") & (d.session == "session1"))[0]
    b = np.flatnonzero((d.subject == "s1") & (d.session == "session2"))[0]
    x[b] = x[a]
    modified = TrialDataset(x, d.y, d.subject, d.session, d.source)
    with pytest.raises(ValueError, match="Split leakage"):
        run_study(modified, protocols=("cross_session",))


def test_invalid_data_and_unknown_protocol_rejected():
    d = synthetic_dataset()
    with pytest.raises(ValueError, match="finite"):
        x = d.X.copy()
        x[0, 0, 0] = np.nan
        TrialDataset(x, d.y, d.subject, d.session, d.source)
    with pytest.raises(ValueError, match="Unknown protocol"):
        build_folds(d, protocols=("pooled_random",))


def test_missing_group_support_is_explicitly_skipped():
    d = synthetic_dataset()
    y = d.y.copy()
    mask = (d.subject == "s1") & (d.session == "session1")
    y[mask] = "left"
    modified = TrialDataset(d.X, y, d.subject, d.session, d.source)
    report = run_study(modified, protocols=("cross_session",))
    assert any(r["status"] == "skipped" for r in report["results"])
    assert any(r["status"] == "evaluated" for r in report["results"])


def test_npz_roundtrip_without_pickle(tmp_path):
    d = synthetic_dataset()
    path = tmp_path / "epochs.npz"
    np.savez_compressed(
        path, X=d.X, y=d.y, subject=d.subject, session=d.session, run=d.run
    )
    loaded = load_npz(path)
    assert np.array_equal(loaded.X, d.X)
    assert loaded.source == "npz:epochs.npz"
    assert np.array_equal(loaded.run, d.run)


def test_public_adapter_validates_subjects_without_download():
    with pytest.raises(ValueError, match="between 1 and 9"):
        load_bnci2014_001([0, 10])


def test_cli_writes_reproducible_json(tmp_path):
    output = tmp_path / "study.json"
    assert main(["--source", "synthetic", "--protocol", "cross_subject", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["summary"]["cross_subject"]["folds"] == 4
    assert payload["schema_version"] == "1.0"
    assert payload["evidence_status"] == "diagnostic_association_only"


def test_known_channel_swap_exposes_conditional_shift_and_transfer_failure():
    gain = run_study(synthetic_dataset(seed=17, shift="gain"), seed=17)
    swapped = run_study(synthetic_dataset(seed=17, shift="channel_swap"), seed=17)
    assert (
        swapped["summary"]["cross_session"]["mean_balanced_accuracy"]
        < gain["summary"]["cross_session"]["mean_balanced_accuracy"] - 0.2
    )
    assert (
        swapped["summary"]["within_session"]["mean_balanced_accuracy"]
        > swapped["summary"]["cross_session"]["mean_balanced_accuracy"] + 0.2
    )
    assert all(
        "class_conditional_shift_z" in r
        for r in swapped["results"]
        if r["status"] == "evaluated"
    )


def test_comparisons_use_same_heldout_session_reference():
    report = run_study(synthetic_dataset(seed=22), seed=22)
    within = {}
    for row in report["results"]:
        if row["status"] == "evaluated" and row["protocol"] == "within_session_runout":
            key = "/".join(row["key"].split("/")[:2])
            within.setdefault(key, []).append(row["balanced_accuracy"])
    for gap in report["paired_within_to_cross_session_gaps"]:
        expected = float(np.mean(within[gap["held_out"]]))
        assert abs(gap["within_session_reference"] - expected) < 1e-5
        assert gap["subject"] == gap["held_out"].split("/")[0]


def test_missing_run_metadata_remains_explicit_and_uses_less_robust_baseline():
    d = synthetic_dataset()
    without_runs = TrialDataset(d.X, d.y, d.subject, d.session, d.source)
    report = run_study(without_runs)
    assert report["summary"]["within_session_runout"]["folds"] == 0
    assert not report["dataset"]["has_run_metadata"]
    assert all(
        gap["reference_protocol"] == "within_session"
        for gap in report["paired_within_to_cross_session_gaps"]
    )


def test_run_metadata_shape_is_validated():
    d = synthetic_dataset()
    with pytest.raises(ValueError, match="run must contain"):
        TrialDataset(d.X, d.y, d.subject, d.session, d.source, run=d.run[:-1])


def test_bootstrap_insufficient_subjects_is_explicit():
    d = synthetic_dataset()
    selected = np.isin(d.subject, ["s1", "s2"])
    limited = TrialDataset(
        d.X[selected], d.y[selected], d.subject[selected],
        d.session[selected], d.source, run=d.run[selected],
    )
    report = run_study(limited)
    assert report["subject_clustered_uncertainty"]["status"] == (
        "insufficient_independent_subjects"
    )
    assert report["subject_clustered_uncertainty"]["ci95"] is None


def test_known_channel_mapping_repair_recovers_generalization():
    baseline = run_study(synthetic_dataset(seed=17, shift="gain"), seed=17)
    affected = run_study(synthetic_dataset(seed=17, shift="channel_swap"), seed=17)
    fixed = run_study(
        repair_known_synthetic_channel_map(
            synthetic_dataset(seed=17, shift="channel_swap")
        ),
        seed=17,
    )
    before = affected["summary"]["cross_session"]["mean_balanced_accuracy"]
    after = fixed["summary"]["cross_session"]["mean_balanced_accuracy"]
    baseline_score = baseline["summary"]["cross_session"]["mean_balanced_accuracy"]
    assert after > before + 0.2
    assert after == baseline_score
    assert fixed["dataset"]["source"].endswith(":oracle_channel_map_repaired")


def test_oracle_repair_refuses_unrelated_data(tmp_path):
    with pytest.raises(ValueError, match="only the channel_swap"):
        repair_known_synthetic_channel_map(synthetic_dataset(shift="gain"))
    with pytest.raises(SystemExit):
        main([
            "--source", "npz", "--npz", str(tmp_path / "missing.npz"),
            "--synthetic-oracle-repair",
        ])
