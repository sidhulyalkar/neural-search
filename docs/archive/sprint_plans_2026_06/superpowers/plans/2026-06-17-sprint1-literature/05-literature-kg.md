# Task 05 — Literature KG Integration

**File to create:** `neural_search/literature/kg_builder.py`
**File to modify:** `scripts/rebuild_full_corpus_graph.py`

---

## New KG Node Types

| Type | Key field | Example |
|------|-----------|---------|
| `paper` | `paper_id` | `W2741809807` |
| `finding` | `finding_id` | `W2741809807:f0` |
| `venue` | `venue_name` | `Nature Neuroscience` |

## New KG Edge Types

| Edge | From | To | Weight |
|------|------|----|--------|
| `paper_reports_finding` | paper | finding | 1.0 |
| `finding_involves_region` | finding | brain_region | confidence |
| `finding_involves_task` | finding | task | confidence |
| `finding_involves_modality` | finding | modality | confidence |
| `finding_involves_species` | finding | species | confidence |
| `dataset_linked_to_paper` | dataset | paper | link confidence |
| `paper_published_in` | paper | venue | 1.0 |

## kg_builder.py Spec

```python
def add_papers_from_shards(
    graph,   # existing KnowledgeGraph
    shard_dir: Path,
    links_path: Path | None = None,
) -> dict:
    """Add paper + venue nodes from JSONL shards.
    
    Returns {"papers_added": N, "venues_added": M, "links_added": K}
    """
    ...

def add_findings_to_graph(
    graph,
    findings_path: Path,
) -> dict:
    """Add finding nodes + edges from findings JSONL.
    
    Returns {"findings_added": N, "edges_added": M}
    """
    ...
```

## rebuild_full_corpus_graph.py Changes

Add two new phases after existing dataset graph build:

```python
# Phase 4: Literature nodes
if literature_shards.exists():
    stats = add_papers_from_shards(graph, literature_shards, links_path)
    print(f"Literature: {stats}")

# Phase 5: Finding nodes + edges  
if findings_path.exists():
    stats = add_findings_to_graph(graph, findings_path)
    print(f"Findings: {stats}")
```

## Tests (tests/test_literature_kg.py)

```python
def test_add_papers_creates_paper_nodes(tmp_path)
def test_add_papers_creates_venue_nodes(tmp_path)
def test_add_papers_dataset_link_edges(tmp_path)
def test_add_findings_creates_finding_nodes(tmp_path)
def test_finding_region_edges(tmp_path)
def test_finding_task_edges(tmp_path)
def test_duplicate_papers_not_added_twice(tmp_path)
```
