# Task 04 — Abstract-to-Finding Extraction

**File to create:** `neural_search/literature/finding_extractor.py`
**File to create:** `scripts/literature/extract_findings.py`
**File to create:** `configs/literature/finding_extraction_v1.yaml`

---

## Goal

For each paper with an abstract, extract 1–3 structured findings using
Claude Haiku. A finding is the atomic scientific claim: what was measured,
in what region/species/modality, and what the result was.

## FindingRecord Schema

```python
@dataclass
class FindingRecord:
    paper_id: str            # OpenAlex ID, e.g. "W2741809807"
    paper_doi: str | None
    finding_id: str          # f"{paper_id}:f{idx}"
    finding_text: str        # 1–2 sentence summary of the finding
    result_direction: str    # "increase" | "decrease" | "no_change" | "correlation" | "mechanism" | "other"
    regions: list[str]       # brain regions mentioned
    species: list[str]
    modalities: list[str]
    tasks: list[str]
    cell_types: list[str]
    molecules: list[str]     # neurotransmitters, receptors, genes mentioned
    confidence: float        # 0.0–1.0 (LLM self-rated)
    extraction_model: str    # e.g. "claude-haiku-4-5-20251001"
    extracted_at: str        # ISO timestamp
```

## Prompt Design

```yaml
# configs/literature/finding_extraction_v1.yaml
model: claude-haiku-4-5-20251001
max_tokens: 512
temperature: 0.0
max_papers_per_run: 10000
batch_size: 50              # papers per API call (use message batching)

system_prompt: |
  You are a neuroscience research assistant. Extract 1–3 key scientific
  findings from the abstract below. For each finding output a JSON object.

  Rules:
  - findings must be specific claims, not background statements
  - result_direction must be one of: increase, decrease, no_change,
    correlation, mechanism, other
  - regions/species/modalities/tasks/cell_types/molecules should be
    canonical lowercase terms, not abbreviations
  - confidence: 0.9 if finding is explicit, 0.6 if inferred
  - if no specific finding, return empty array []

  Output format:
  ```json
  [
    {
      "finding_text": "...",
      "result_direction": "...",
      "regions": [...],
      "species": [...],
      "modalities": [...],
      "tasks": [...],
      "cell_types": [...],
      "molecules": [...],
      "confidence": 0.9
    }
  ]
  ```

user_template: |
  Title: {title}
  Abstract: {abstract}
```

## finding_extractor.py Spec

```python
def load_config(path: Path) -> dict: ...

def build_prompt(title: str, abstract: str, config: dict) -> str: ...

def parse_findings(
    paper_id: str,
    paper_doi: str | None,
    response_text: str,
    model: str,
) -> list[FindingRecord]:
    """Parse JSON array from LLM response. Returns [] on parse failure."""
    ...

def extract_batch(
    papers: list[dict],   # [{paper_id, paper_doi, title, abstract}]
    config: dict,
    *,
    api_key: str | None = None,
) -> list[FindingRecord]:
    """Send one paper per message to Haiku. Returns all findings."""
    ...

def extract_findings_from_corpus(
    corpus_shards: list[Path],
    config: dict,
    out_path: Path,
    *,
    checkpoint_path: Path | None = None,
    max_papers: int | None = None,
) -> int:
    """Process all shards. Checkpoint after each shard. Returns count."""
    ...
```

## CLI Spec

```
python scripts/literature/extract_findings.py \
    --corpus data/corpus/normalized/openalex_neuro \
    --config configs/literature/finding_extraction_v1.yaml \
    --out artifacts/literature/findings_v1.jsonl \
    [--max-papers 10000] \
    [--resume]
```

## Graceful degradation
- If `ANTHROPIC_API_KEY` not set: exits with code 0, prints warning
- If API call fails: logs error, skips paper, continues
- If JSON parse fails: logs warning, `finding_text` = raw response[:200], confidence = 0.0

## Tests (tests/test_finding_extractor.py)

```python
class TestParseFindings:
    def test_valid_json_array()
    def test_empty_array()
    def test_malformed_json_returns_empty()
    def test_finding_ids_are_unique()
    def test_confidence_clamped_to_01()

class TestBuildPrompt:
    def test_includes_title_and_abstract()
    def test_uses_config_system_prompt()

class TestExtractBatch:
    def test_no_api_key_returns_empty(monkeypatch)
    def test_skips_papers_without_abstract()
```

## Output format (JSONL)

```json
{"paper_id": "W2741809807", "paper_doi": "10.1038/s41593-020-0636-4",
 "finding_id": "W2741809807:f0",
 "finding_text": "Neuropixels recordings in mouse visual cortex reveal that ...",
 "result_direction": "correlation",
 "regions": ["visual cortex", "v1"],
 "species": ["mouse"],
 "modalities": ["neuropixels"],
 "tasks": [],
 "cell_types": ["pyramidal neuron"],
 "molecules": [],
 "confidence": 0.9,
 "extraction_model": "claude-haiku-4-5-20251001",
 "extracted_at": "2026-06-17T12:00:00Z"}
```
