import json
from pathlib import Path

from neural_search.field_state.cli import main
from neural_search.field_state.eval_memory.adjudication import adjudicate_review_group
from neural_search.field_state.eval_memory.qrels_import import parse_human_label_fields
from neural_search.field_state.eval_memory.qrels_schema import (
    AdjudicatedQrel,
    QrelsCandidate,
    QrelsReview,
    stable_qrels_candidate_id,
)
from neural_search.field_state.obsidian.frontmatter import (
    compose_note,
    parse_frontmatter,
)
from neural_search.field_state.obsidian.templates import HUMAN_BEGIN, HUMAN_END
from neural_search.field_state.store import (
    ADJUDICATED_QRELS_PATH,
    CLAIM_EVIDENCE_SUGGESTIONS_PATH,
    EVAL_SNAPSHOT_PATH,
    QRELS_AGREEMENT_PATH,
    QRELS_CANDIDATES_PATH,
    QRELS_REVIEWS_PATH,
    WHITEPAPER_VALIDATION_REPORT,
    write_jsonl,
)


def _write_fixture_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    pool = _write_fixture_jsonl(
        tmp_path / "fixtures/benchmark_pool.jsonl",
        [
            {
                "query_id": "q_1",
                "record_id": "dandi:000001",
                "pooled_from": ["bm25", "usefulness"],
                "min_rank": 1,
                "status": "needs_annotation",
            }
        ],
    )
    queries = _write_fixture_jsonl(
        tmp_path / "fixtures/benchmark_queries.jsonl",
        [
            {
                "query_id": "q_1",
                "intent": "PIPELINE_REUSE",
                "query": "mouse neuropixels spike sorting",
                "known_failure_modes": ["calcium imaging only"],
            }
        ],
    )
    corpus = _write_fixture_jsonl(
        tmp_path / "fixtures/combined_corpus.jsonl",
        [
            {
                "source": "dandi",
                "source_id": "000001",
                "title": "Mouse Neuropixels dataset",
                "description": "Awake behaving mouse electrophysiology.",
            }
        ],
    )
    return pool, queries, corpus


def _export_fixture_qrels(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    pool, queries, corpus = _fixtures(tmp_path)
    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "qrels-export",
                "--pool",
                str(pool),
                "--queries",
                str(queries),
                "--corpus",
                str(corpus),
                "--vault",
                str(vault),
                "--limit",
                "10",
            ]
        )
        == 0
    )
    return vault


def _qrels_note(vault: Path) -> Path:
    return next((vault / "Field-State/60_Evaluation/qrels_review/unreviewed").glob("*.md"))


def test_qrels_candidate_schema_and_stable_id():
    candidate_id = stable_qrels_candidate_id("q_1", "dandi:000001")
    candidate = QrelsCandidate(
        id=candidate_id,
        query_id="q_1",
        query_text="mouse neuropixels",
        dataset_id="dandi:000001",
        dataset_title="Mouse dataset",
    )

    assert candidate.id == "qrels_candidate:q_1:dandi:000001"
    assert QrelsCandidate.from_jsonl(candidate.to_jsonl()).dataset_id == "dandi:000001"


def test_qrels_review_and_adjudicated_schema():
    review = QrelsReview(
        candidate_id="qrels_candidate:q_1:dandi:000001",
        query_id="q_1",
        dataset_id="dandi:000001",
        relevance_score=3,
        usefulness_score=2,
        hard_negative_violation=False,
        review_status="reviewed",
    )
    qrel = AdjudicatedQrel(
        candidate_id=review.candidate_id,
        query_id=review.query_id,
        dataset_id=review.dataset_id,
        final_relevance_score=3,
        final_hard_negative_violation=False,
        adjudication_status="single_review",
    )

    assert QrelsReview.from_jsonl(review.to_jsonl()).relevance_score == 3
    assert AdjudicatedQrel.from_jsonl(qrel.to_jsonl()).adjudication_status == "single_review"


def test_qrels_export_creates_candidates_and_notes(tmp_path: Path):
    vault = _export_fixture_qrels(tmp_path)

    assert (tmp_path / QRELS_CANDIDATES_PATH).exists()
    note = _qrels_note(vault)
    assert "Qrels Review" in note.read_text(encoding="utf-8")
    assert "mouse neuropixels spike sorting" in note.read_text(encoding="utf-8")


def test_parse_human_label_fields():
    fields = parse_human_label_fields(
        "relevance_score: 3\nusefulness_score: 2\nhard_negative_violation: false\n"
    )

    assert fields["relevance_score"] == "3"
    assert fields["hard_negative_violation"] == "false"


def test_qrels_import_writes_reviews_jsonl(tmp_path: Path):
    vault = _export_fixture_qrels(tmp_path)
    note = _qrels_note(vault)
    text = note.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    human_block = (
        f"{HUMAN_BEGIN}\n\n"
        "## Human label\n\n"
        "relevance_score: 3\n"
        "usefulness_score: 2\n"
        "hard_negative_violation: false\n"
        "label_confidence: high\n"
        "annotator_id: reviewer_a\n\n"
        "## Rationale\n\n"
        "Strong modality and task match.\n\n"
        "## Reviewer notes\n\n"
        "Looks reusable.\n\n"
        f"{HUMAN_END}"
    )
    old_human = body[body.index(HUMAN_BEGIN) : body.index(HUMAN_END) + len(HUMAN_END)]
    frontmatter["review_status"] = "reviewed"
    note.write_text(compose_note(frontmatter, body.replace(old_human, human_block)), encoding="utf-8")

    assert main(["--root", str(tmp_path), "qrels-import", "--vault", str(vault)]) == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / QRELS_REVIEWS_PATH).read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["relevance_score"] == 3
    assert rows[0]["annotator_id"] == "reviewer_a"


def test_adjudication_single_agreement_and_disagreement(tmp_path: Path):
    base = QrelsReview(
        candidate_id="qrels_candidate:q_1:dandi:000001",
        query_id="q_1",
        dataset_id="dandi:000001",
        relevance_score=2,
        hard_negative_violation=False,
        review_status="reviewed",
    )
    single = adjudicate_review_group([base])
    agreement = adjudicate_review_group([base, base.model_copy(update={"annotator_id": "b"})])
    disagreement = adjudicate_review_group(
        [base, base.model_copy(update={"annotator_id": "b", "relevance_score": 1})]
    )

    assert single is not None and single.adjudication_status == "single_review"
    assert agreement is not None and agreement.adjudication_status == "agreement"
    assert disagreement is not None and disagreement.adjudication_status == "needs_adjudication"

    write_jsonl(QRELS_REVIEWS_PATH, [base], tmp_path)
    assert main(["--root", str(tmp_path), "qrels-adjudicate"]) == 0
    assert (tmp_path / ADJUDICATED_QRELS_PATH).exists()


def test_qrels_agreement_eval_snapshot_claim_evidence_and_whitepaper(tmp_path: Path):
    review_a = QrelsReview(
        candidate_id="qrels_candidate:q_1:dandi:000001",
        query_id="q_1",
        dataset_id="dandi:000001",
        annotator_id="a",
        relevance_score=3,
        usefulness_score=3,
        hard_negative_violation=False,
        review_status="reviewed",
    )
    review_b = review_a.model_copy(update={"annotator_id": "b"})
    assert main(["--root", str(tmp_path), "init"]) == 0
    write_jsonl(QRELS_CANDIDATES_PATH, [], tmp_path)
    write_jsonl(QRELS_REVIEWS_PATH, [review_a, review_b], tmp_path)

    assert main(["--root", str(tmp_path), "qrels-adjudicate"]) == 0
    assert main(["--root", str(tmp_path), "qrels-agreement"]) == 0
    assert main(["--root", str(tmp_path), "eval-snapshot"]) == 0
    assert main(["--root", str(tmp_path), "claim-evidence-update"]) == 0
    assert main(["--root", str(tmp_path), "whitepaper-validation-report"]) == 0

    agreement = json.loads((tmp_path / QRELS_AGREEMENT_PATH).read_text(encoding="utf-8"))
    snapshot = json.loads((tmp_path / EVAL_SNAPSHOT_PATH).read_text(encoding="utf-8"))
    assert agreement["exact_agreement_rate"] == 1.0
    assert snapshot["qrels_status"]["candidates_adjudicated"] == 1
    assert (tmp_path / CLAIM_EVIDENCE_SUGGESTIONS_PATH).exists()
    assert (tmp_path / WHITEPAPER_VALIDATION_REPORT).exists()


def test_existing_field_state_and_obsidian_commands_still_work(tmp_path: Path):
    vault = tmp_path / "vault"
    assert main(["--root", str(tmp_path), "init"]) == 0
    assert main(["--root", str(tmp_path), "report"]) == 0
    assert main(["--root", str(tmp_path), "opportunities"]) == 0
    assert main(["--root", str(tmp_path), "snapshot"]) == 0
    assert main(["--root", str(tmp_path), "export-obsidian", "--vault", str(vault)]) == 0
    assert main(["memory-validate", "--vault", str(vault)]) == 0
