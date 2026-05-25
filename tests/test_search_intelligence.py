from __future__ import annotations

from pathlib import Path

import yaml

from neural_search.intelligence import (
    EvaluationQuery,
    apply_search_intelligence_config,
    build_benchmark_query_seeds,
    build_review_queue,
    build_search_coverage_plan,
    evaluate_promotion_gates,
    evaluate_query_plan,
    load_relevance_judgments,
    load_search_records_from_normalized,
    plan_search_intelligence,
    run_query_plan_evaluation,
    search_datasets_with_intelligence,
    summarize_relevance_judgments,
    write_promotion_gate_report,
    write_query_plan_evaluation_report,
    write_review_queue,
    write_search_coverage_plan,
)
from neural_search.intelligence.evaluation import main as evaluation_main
from neural_search.intelligence.planner import main as planner_main
from neural_search.intelligence.promotion import main as promotion_main
from neural_search.intelligence.review import main as review_main
from neural_search.normalized import make_dataset_id, write_jsonl
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=f"label:{label_type}:{label}",
        label=label,
        label_type=label_type,
        confidence=0.95,
    )


def _search_record(
    source_id: str,
    title: str,
    description: str,
    *,
    species: list[str],
    modalities: list[str],
    behaviors: list[str] | None = None,
    tasks: list[str] | None = None,
) -> dict:
    return {
        "dataset": {
            "id": source_id,
            "source": "fixture",
            "source_id": source_id,
            "title": title,
            "description": description,
            "species": species,
            "modalities": modalities,
            "brain_regions": [],
            "tasks": tasks or [],
            "behaviors": behaviors or [],
            "data_standards": ["BIDS"],
            "has_behavior": bool(behaviors),
            "has_trials": True,
            "license": "CC0",
            "linked_paper_ids": [],
            "metadata_json": {},
        }
    }


def test_search_intelligence_plan_handles_cross_modal_hard_negatives() -> None:
    plan = plan_search_intelligence(
        "human EEG BCI decoding with behavior without fMRI",
        corpus_profile={
            "data_form_counts": {"eeg_meg": 0, "behavior": 4},
            "underrepresented_data_forms": ["eeg_meg"],
        },
    )

    assert plan.intent == "hard_negative"
    assert plan.mode == "constraint_filter_first"
    assert "eeg_meg" in plan.required_data_forms
    assert "mri" in plan.excluded_data_forms
    assert "preserve_hard_negative_filtering" in plan.quality_checks
    assert "awareness" in plan.retrieval_weights
    assert any("eeg_meg" in warning for warning in plan.warnings)


def test_search_intelligence_plan_promotes_cross_modal_fit() -> None:
    plan = plan_search_intelligence(
        "mouse Neuropixels and calcium imaging population dynamics"
    )

    assert plan.intent == "cross_modal"
    assert plan.mode == "cross_modal_fit"
    assert "extracellular_ephys" in plan.required_data_forms
    assert "optical_imaging" in plan.required_data_forms
    assert "verify_cross_modal_alignment" in plan.quality_checks
    assert plan.retrieval_weights["awareness"] >= 0.2


def test_coverage_plan_prioritizes_missing_data_forms(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    benchmark_path = tmp_path / "benchmark.yaml"
    output_dir = tmp_path / "reports"
    record = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("fixture", "001"),
        source="fixture",
        source_id="001",
        title="Mouse Neuropixels decision behavior",
        description="NWB units, spike times, events, and behavior trials.",
        species=[_label("species", "mouse")],
        modalities=[_label("modality", "neuropixels")],
        behavioral_events=[_label("behavior", "choice")],
        data_standards=[_label("data_standard", "NWB")],
    )
    write_jsonl([record], records_path)
    benchmark_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_queries": [
                    {
                        "id": "q1",
                        "query": "mouse Neuropixels decoding without EEG",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = build_search_coverage_plan(
        records_path,
        benchmark_path,
        target_corpus_count=1,
        target_benchmark_query_count=1,
    )
    paths = write_search_coverage_plan(plan, output_dir)

    gap_ids = {gap.data_form for gap in plan.gaps}
    assert "mri" in gap_ids
    assert "molecular" in gap_ids
    assert "extracellular_ephys" not in gap_ids
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    assert Path(paths["benchmark_seeds"]).exists()


def test_coverage_plan_generates_reviewable_benchmark_seeds(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    write_jsonl([], records_path)

    plan = build_search_coverage_plan(
        records_path,
        target_corpus_count=1,
        target_benchmark_query_count=1,
    )
    seeds = build_benchmark_query_seeds(plan, max_gaps=1, queries_per_gap=1)

    assert seeds["metadata"]["review_required"] is True
    assert seeds["benchmark_queries"][0]["coverage_gap"]
    assert seeds["benchmark_queries"][0]["minimum_precision_at_5"] == 0.0


def test_planner_cli_writes_query_plan(tmp_path: Path) -> None:
    output_path = tmp_path / "plan.json"
    exit_code = planner_main(
        [
            "--query",
            "connectomics morphology excluding fMRI",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = output_path.read_text(encoding="utf-8")
    assert "connectomics" in payload
    assert "constraint_filter_first" in payload


def test_intelligence_config_application_preserves_hard_negative_filters() -> None:
    application = apply_search_intelligence_config(
        "human EEG without fMRI",
        {
            "weights": {"ontology": 0.2, "semantic": 0.2},
            "intelligence": {"enabled": True, "weight_blend_strength": 0.5},
        },
    )

    assert application.enabled is True
    assert application.plan.intent == "hard_negative"
    assert application.retrieval_config["hard_negative_filters"]["enabled"] is True
    assert application.blended_weights["negative_constraint"] > 0


def test_search_with_intelligence_exposes_plan_metadata() -> None:
    response = search_datasets_with_intelligence(
        "human EEG BCI decoding with behavior without fMRI",
        datasets=[
            _search_record(
                "GOOD_EEG",
                "Human EEG BCI motor imagery",
                "Human EEG channels, events, labels, sampling rate, and behavior trials.",
                species=["human"],
                modalities=["eeg"],
                behaviors=["motor imagery"],
                tasks=["bci_decoding"],
            ),
            _search_record(
                "BAD_FMRI",
                "Human fMRI behavior task",
                "Human BOLD fMRI images and behavior events.",
                species=["human"],
                modalities=["fmri"],
                behaviors=["button press"],
                tasks=["decision making"],
            ),
        ],
        limit=2,
        rerank=True,
    )

    assert response.parsed_query["search_intelligence_enabled"] is True
    assert response.parsed_query["search_intelligence_plan"]["intent"] == "hard_negative"
    assert response.results[0].dataset_id == "GOOD_EEG"
    assert "awareness_score" in response.results[0].score_breakdown


def test_relevance_judgments_and_review_queue(tmp_path: Path) -> None:
    judgments_path = tmp_path / "judgments.jsonl"
    judgments_path.write_text(
        "\n".join(
            [
                (
                    '{"query_id":"q1","query_text":"eeg","dataset_id":"GOOD",'
                    '"relevance":"exact","confidence":0.9}'
                ),
                (
                    '{"query_id":"q1","query_text":"eeg","dataset_id":"BAD",'
                    '"relevance":"hard_negative","confidence":0.8}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    judgments = load_relevance_judgments(judgments_path)
    summary = summarize_relevance_judgments(judgments)
    queue = build_review_queue(
        {
            "gaps": [
                {"data_form": "molecular", "priority": "critical"},
            ]
        },
        {
            "benchmark_queries": [
                {
                    "id": "coverage_molecular_01",
                    "query": "single cell datasets",
                    "coverage_gap": "molecular",
                    "priority": "critical",
                }
            ]
        },
    )
    paths = write_review_queue(queue, tmp_path / "review")

    assert summary["judgment_count"] == 2
    assert summary["positive_count"] == 1
    assert summary["hard_negative_count"] == 1
    assert queue[0]["label_status"] == "needs_review"
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()


def test_review_queue_cli_writes_reports(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.json"
    seeds_path = tmp_path / "seeds.yaml"
    out_dir = tmp_path / "queue"
    coverage_path.write_text(
        '{"gaps":[{"data_form":"connectomics","priority":"critical"}]}',
        encoding="utf-8",
    )
    seeds_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_queries": [
                    {
                        "id": "coverage_connectomics_01",
                        "query": "connectome datasets",
                        "coverage_gap": "connectomics",
                        "priority": "critical",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = review_main(
        [
            "--coverage",
            str(coverage_path),
            "--benchmark-seeds",
            str(seeds_path),
            "--out",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "human_review_queue.json").exists()


def test_query_plan_evaluation_compares_variants(tmp_path: Path) -> None:
    datasets = [
        _search_record(
            "GOOD_EEG",
            "Human EEG BCI motor imagery",
            "Human EEG channels, events, labels, sampling rate, and behavior trials.",
            species=["human"],
            modalities=["eeg"],
            behaviors=["motor imagery"],
            tasks=["bci_decoding"],
        ),
        _search_record(
            "BAD_FMRI",
            "Human fMRI behavior task",
            "Human BOLD fMRI images and behavior events.",
            species=["human"],
            modalities=["fmri"],
            behaviors=["button press"],
            tasks=["decision making"],
        ),
    ]
    query = EvaluationQuery(
        id="q_eeg",
        query="human EEG BCI decoding with behavior without fMRI",
        expected_dataset_ids=("GOOD_EEG",),
        hard_negative_dataset_ids=("BAD_FMRI",),
    )

    evaluation = evaluate_query_plan(query, datasets=datasets, limit=2)
    report = run_query_plan_evaluation([query], datasets=datasets, limit=2)
    paths = write_query_plan_evaluation_report(report, tmp_path / "eval")

    assert evaluation.planner_intent == "hard_negative"
    assert evaluation.intelligence.result_ids[0] == "GOOD_EEG"
    assert evaluation.intelligence_delta["hard_negative_violations"] == 0.0
    assert report.promotion_safe is True
    assert report.grouped_by_intent["hard_negative"]["promotion_safe"] is True
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()


def test_query_plan_evaluation_loads_normalized_records(tmp_path: Path) -> None:
    records_path = tmp_path / "normalized.jsonl"
    record = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("fixture", "eeg001"),
        source="fixture",
        source_id="eeg001",
        title="Human BIDS EEG BCI",
        description="BIDS EEG channels, events, sampling rate, and behavior labels.",
        species=[_label("species", "human")],
        modalities=[_label("modality", "eeg")],
        tasks=[_label("task", "motor_imagery")],
        behavioral_events=[_label("behavior", "button_press")],
        data_standards=[_label("data_standard", "BIDS")],
    )
    write_jsonl([record], records_path)

    datasets = load_search_records_from_normalized(records_path)
    report = run_query_plan_evaluation(
        [
            EvaluationQuery(
                id="q_eeg",
                query="human EEG motor imagery",
                expected_dataset_ids=(record.dataset_id,),
            )
        ],
        datasets=datasets,
        corpus_label=str(records_path),
        limit=3,
    )

    assert datasets[0]["dataset"]["id"] == record.dataset_id
    assert report.corpus["record_count"] == 1
    assert report.queries[0].intelligence.hit_at_5 == 1.0


def test_query_plan_evaluation_cli_writes_reports(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.yaml"
    out_dir = tmp_path / "report"
    benchmark_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_queries": [
                    {
                        "id": "q1",
                        "query": "mouse Neuropixels decision making",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = evaluation_main(
        [
            "--benchmark",
            str(benchmark_path),
            "--out",
            str(out_dir),
            "--limit",
            "3",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "query_plan_evaluation.json").exists()


def test_promotion_gates_block_until_manifest_and_counts_allow(tmp_path: Path) -> None:
    evaluation_report = {
        "query_count": 3,
        "mean_delta": {"hard_negative_violations": 0},
        "grouped_by_intent": {
            "hard_negative": {
                "query_count": 3,
                "mean_mrr_delta": 0.1,
                "hard_negative_violation_delta": 0,
            }
        },
    }
    manifest = {
        "default_enabled": False,
        "global_gates": {
            "min_total_queries": 5,
            "max_hard_negative_violation_delta": 0,
        },
        "intents": {
            "hard_negative": {
                "enabled": False,
                "min_query_count": 5,
                "min_mean_mrr_delta": 0.0,
                "max_hard_negative_violation_delta": 0,
            }
        },
    }

    report = evaluate_promotion_gates(evaluation_report, manifest)
    paths = write_promotion_gate_report(report, tmp_path / "promotion")

    assert report.promotion_ready is False
    assert "default promotion disabled in manifest" in report.blockers
    assert report.intent_decisions[0].ready is False
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()


def test_promotion_gates_use_human_label_summary() -> None:
    evaluation_report = {
        "query_count": 10,
        "mean_delta": {"hard_negative_violations": 0},
        "grouped_by_intent": {},
    }
    manifest = {
        "default_enabled": True,
        "global_gates": {
            "min_total_queries": 10,
            "max_hard_negative_violation_delta": 0,
        },
        "human_label_gates": {
            "min_judgments": 2,
            "max_hard_negative_count": 0,
        },
        "intents": {},
    }

    report = evaluate_promotion_gates(
        evaluation_report,
        manifest,
        human_label_summary={"judgment_count": 1, "hard_negative_count": 0},
    )

    assert report.promotion_ready is False
    assert "human judgment_count 1 < required 2" in report.blockers


def test_promotion_cli_writes_gate_report(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    manifest_path = tmp_path / "manifest.yaml"
    out_dir = tmp_path / "promotion"
    evaluation_path.write_text(
        '{"query_count":1,"mean_delta":{"hard_negative_violations":0},'
        '"grouped_by_intent":{}}',
        encoding="utf-8",
    )
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "default_enabled": False,
                "global_gates": {"min_total_queries": 1},
                "intents": {},
            }
        ),
        encoding="utf-8",
    )

    exit_code = promotion_main(
        [
            "--manifest",
            str(manifest_path),
            "--evaluation",
            str(evaluation_path),
            "--out",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "promotion_gate_report.json").exists()
