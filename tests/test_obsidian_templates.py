from neural_search.obsidian.templates import (
    claim_card_body,
    claim_card_frontmatter,
    paper_card_body,
    paper_card_frontmatter,
)

SAMPLE_PAPER = {
    "paper_id": "paper:openalex:W123",
    "doi": "10.1234/example",
    "title": "Theta oscillations in CA1",
    "authors": ["Buzsaki G"],
    "year": 2021,
    "n_findings": 3,
    "finding_ids": ["f1", "f2", "f3"],
    "linked_datasets": ["dandi:000026"],
    "modalities": ["neuropixels"],
    "regions": ["hippocampus"],
    "species": ["mouse"],
    "extraction_model": "claude-haiku-4-5-20251001",
    "extraction_prompt_version": "extraction_v2",
}

SAMPLE_CLAIM = {
    "claim_id": "node:claim:hippocampus_increase_abc12345",
    "statement": "Theta oscillations increase during spatial navigation",
    "direction": "increase",
    "regions": ["hippocampus"],
    "species": ["mouse"],
    "consensus_confidence": 0.87,
    "n_supporting_findings": 5,
    "n_contradicting_findings": 0,
    "magnitude_summary": "r=0.7",
    "timescale": "millisecond",
    "evidence_strength": "direct",
    "status": "active",
    "supporting_datasets": ["dandi:000026"],
    "supporting_papers": ["paper:openalex:W123"],
    "contradicted_by": [],
    "synthesis_model": "claude-haiku-4-5-20251001",
    "synthesis_prompt_version": "synthesis_v1",
    "synthesized_at": "2026-06-21T00:00:00+00:00",
    "agent_digest": "5 findings show theta increases during spatial nav.",
}


def test_paper_card_frontmatter_has_required_fields():
    fm = paper_card_frontmatter(SAMPLE_PAPER)
    assert fm["paper_id"] == "paper:openalex:W123"
    assert fm["type"] == "paper"
    assert fm["n_findings"] == 3
    assert "dandi:000026" in fm["linked_datasets"]


def test_claim_card_frontmatter_has_required_fields():
    fm = claim_card_frontmatter(SAMPLE_CLAIM)
    assert fm["claim_id"] == "node:claim:hippocampus_increase_abc12345"
    assert fm["type"] == "claim"
    assert fm["status"] == "active"
    assert fm["consensus_confidence"] == 0.87


def test_claim_card_body_includes_agent_digest():
    body = claim_card_body(SAMPLE_CLAIM)
    assert "Agent Digest" in body
    assert "5 findings show theta increases" in body


def test_paper_card_body_includes_title():
    body = paper_card_body(SAMPLE_PAPER)
    assert "Theta oscillations in CA1" in body
