# Query Intent and Field Routing

This document defines how natural language scientific queries should be parsed into retrieval intents, constraints, and field-specific search routes for Neural Search v0.4.

## Query Intent Categories

### 1. dataset_lookup

**Description:** Direct lookup of a specific dataset by ID, name, or source.

**Examples:**
- "Find DANDI dataset 000026"
- "Get the Neuropixels visual coding dataset"
- "OpenNeuro ds003505"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| lexical | 0.50 |
| ontology | 0.20 |
| semantic | 0.20 |
| provenance | 0.05 |
| usability | 0.05 |

**Fields to Search:**
- title (primary)
- source_id
- dataset_id
- description (secondary)

**Routing Logic:**
- If query contains "DANDI", "OpenNeuro", dataset IDs → route to dataset_lookup
- Exact match on source_id gets maximum boost

---

### 2. paper_lookup

**Description:** Direct lookup of a specific paper by title, DOI, or author.

**Examples:**
- "Find the Steinmetz 2019 Neuropixels paper"
- "Paper with DOI 10.1038/s41586-019-1787-x"
- "Papers by Churchland on motor cortex"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| lexical | 0.45 |
| semantic | 0.30 |
| ontology | 0.15 |
| provenance | 0.10 |

**Fields to Search:**
- title (primary)
- abstract
- authors
- doi
- year

**Routing Logic:**
- If query contains "paper", "publication", DOI pattern, author names → route to paper_lookup

---

### 3. task_search

**Description:** Find datasets/papers by cognitive or behavioral task.

**Examples:**
- "Go/no-go task datasets"
- "Reversal learning experiments"
- "Motor imagery BCI paradigms"
- "Visual decision-making tasks"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| ontology | 0.40 |
| semantic | 0.25 |
| analysis_fit | 0.20 |
| lexical | 0.10 |
| usability | 0.05 |

**Fields to Search:**
- tasks (primary, exact match)
- behavioral_events
- analysis_goals
- title
- description

**Routing Logic:**
- If query contains known task terms from ontology → route to task_search
- Expand task terms to related behavioral events

---

### 4. modality_region_species_search

**Description:** Find datasets by recording modality, brain region, or species.

**Examples:**
- "Neuropixels recordings in mouse V1"
- "Human ECoG motor cortex"
- "Calcium imaging hippocampus"
- "Macaque electrophysiology prefrontal cortex"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| ontology | 0.45 |
| semantic | 0.20 |
| lexical | 0.15 |
| usability | 0.10 |
| analysis_fit | 0.10 |

**Fields to Search:**
- modalities (primary)
- brain_regions (primary)
- species (primary)
- title
- description

**Routing Logic:**
- If query contains modality + region or species → route to modality_region_species_search
- Apply strict filtering on species when specified

---

### 5. analysis_affordance_search

**Description:** Find datasets suitable for specific analysis types.

**Examples:**
- "Datasets for choice decoding"
- "Data suitable for Q-learning modeling"
- "Recordings with event-aligned neural activity"
- "Datasets I can use for state-space modeling"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| analysis_fit | 0.45 |
| ontology | 0.25 |
| usability | 0.15 |
| semantic | 0.10 |
| provenance | 0.05 |

**Fields to Search:**
- analysis_affordances (primary)
- analysis_goals
- tasks
- behavioral_events
- usability_flags

**Routing Logic:**
- If query contains "for [analysis]", "suitable for", "can support" → route to analysis_affordance_search
- Check analysis affordance support levels

---

### 6. paper_to_dataset_linking

**Description:** Find datasets associated with a specific paper.

**Examples:**
- "What datasets did Steinmetz 2019 use?"
- "Data from the IBL brain-wide map paper"
- "Datasets linked to this DOI"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| provenance | 0.50 |
| lexical | 0.25 |
| semantic | 0.15 |
| ontology | 0.10 |

**Fields to Search:**
- linked_papers
- paper_id references
- title overlap
- author overlap

**Routing Logic:**
- If query references a paper and asks for "data", "datasets" → route to paper_to_dataset_linking

---

### 7. dataset_to_paper_linking

**Description:** Find papers that used or describe a specific dataset.

**Examples:**
- "Papers using DANDI 000026"
- "Publications about the Allen Visual Coding dataset"
- "Studies that analyzed this dataset"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| provenance | 0.50 |
| lexical | 0.25 |
| semantic | 0.15 |
| ontology | 0.10 |

**Fields to Search:**
- linked_datasets
- dataset_id references
- title overlap

**Routing Logic:**
- If query references a dataset and asks for "papers", "publications", "studies" → route to dataset_to_paper_linking

---

### 8. similar_dataset_search

**Description:** Find datasets similar to a given dataset.

**Examples:**
- "Datasets similar to DANDI 000026"
- "Other mouse V1 Neuropixels datasets"
- "More data like this one"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| semantic | 0.35 |
| ontology | 0.30 |
| analysis_fit | 0.20 |
| usability | 0.10 |
| provenance | 0.05 |

**Fields to Search:**
- All label fields from reference dataset
- combined_scientific_summary embedding similarity

**Routing Logic:**
- If query contains "similar to", "like", "more" + dataset reference → route to similar_dataset_search
- Use reference dataset's labels as query expansion

---

### 9. negative_constraint_search

**Description:** Find datasets while excluding specific modalities, species, or tasks.

**Examples:**
- "Mouse electrophysiology, NOT calcium imaging"
- "Decision-making datasets without fMRI"
- "Neural recordings excluding human subjects"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| ontology | 0.35 |
| negative_constraint | 0.30 |
| semantic | 0.20 |
| analysis_fit | 0.10 |
| usability | 0.05 |

**Fields to Search:**
- Standard fields with exclusion filters
- modalities (for exclusion)
- species (for exclusion)
- tasks (for exclusion)

**Routing Logic:**
- If query contains "NOT", "without", "excluding", "no [modality/species]" → route to negative_constraint_search
- Apply hard exclusion penalty for violated constraints

---

### 10. ambiguous_exploratory_search

**Description:** Open-ended exploration without clear constraints.

**Examples:**
- "Interesting neural datasets"
- "What neuroscience data is available?"
- "Show me some example datasets"

**Score Head Weights:**
| Head | Weight |
|------|--------|
| usability | 0.30 |
| provenance | 0.25 |
| semantic | 0.25 |
| ontology | 0.15 |
| analysis_fit | 0.05 |

**Fields to Search:**
- All fields with equal weight
- Prioritize high-quality, well-documented datasets

**Routing Logic:**
- If no specific constraints detected → route to ambiguous_exploratory_search
- Boost datasets with high usability scores

---

## Negative Constraint Parsing

### Patterns to Detect

```python
EXCLUSION_PATTERNS = [
    r"not\s+using\s+(\w+)",
    r"NOT\s+(\w+)",
    r"not\s+(\w+)",
    r"without\s+(\w+)",
    r"excluding\s+(\w+)",
    r"no\s+(\w+)\s+(?:data|recordings?|datasets?)",
    r"exclude\s+(\w+)",
    r"non-(\w+)",  # non-human, non-invasive
]
```

### Constraint Representation

```python
@dataclass
class QueryConstraints:
    required_modalities: list[str]
    required_species: list[str]
    required_tasks: list[str]
    required_regions: list[str]
    required_affordances: list[str]

    excluded_modalities: list[str]
    excluded_species: list[str]
    excluded_tasks: list[str]

    # Hard constraints that must be satisfied
    hard_exclusions: list[str]

    # Soft preferences that boost/penalize
    soft_preferences: dict[str, float]
```

### Exclusion Penalty Application

```python
def apply_exclusion_penalty(
    result: SearchResult,
    constraints: QueryConstraints,
    penalty_weight: float = 0.5
) -> SearchResult:
    """Apply strong penalty for violated exclusions."""

    violated = set()

    # Check modality exclusions
    result_modalities = normalize_labels(result.modalities)
    for excluded in constraints.excluded_modalities:
        if excluded in result_modalities:
            violated.add(f"modality:{excluded}")

    # Check species exclusions
    result_species = normalize_labels(result.species)
    for excluded in constraints.excluded_species:
        if excluded in result_species:
            violated.add(f"species:{excluded}")

    if violated:
        result.negative_constraint_score = 0.0
        result.warnings.append(
            f"Exclusion violation: {', '.join(violated)}"
        )

    return result
```

---

## Field-Specific Embedding Strategy

### Dataset Fields

| Field | Embedding Priority | Use Case |
|-------|-------------------|----------|
| title | High | Direct lookup, semantic similarity |
| description | High | Exploratory search, context |
| tasks | High | Task-specific search |
| behavioral_events | Medium | Event-aligned analysis search |
| modalities | Medium | Modality filtering |
| brain_regions | Medium | Region-specific search |
| analysis_goals | High | Affordance matching |
| combined_scientific_summary | High | General semantic search |

### Paper Fields

| Field | Embedding Priority | Use Case |
|-------|-------------------|----------|
| title | High | Direct lookup |
| abstract | High | Semantic similarity, methods |
| extracted_labels | Medium | Structured matching |
| combined_scientific_summary | High | General semantic search |

### Query-to-Field Routing

```python
INTENT_TO_FIELDS = {
    "dataset_lookup": ["title", "description"],
    "paper_lookup": ["title", "abstract"],
    "task_search": ["tasks", "behavioral_events", "combined_scientific_summary"],
    "modality_region_species_search": ["modalities", "brain_regions", "title"],
    "analysis_affordance_search": ["analysis_goals", "combined_scientific_summary"],
    "similar_dataset_search": ["combined_scientific_summary"],
    "negative_constraint_search": ["combined_scientific_summary"],
    "ambiguous_exploratory_search": ["combined_scientific_summary", "title"],
}
```

---

## Fallback Behavior

### When Intent is Unclear

1. Default to `ambiguous_exploratory_search`
2. Apply balanced weights across all score heads
3. Boost well-documented, high-usability datasets
4. Include diversity in results (different modalities, tasks)

### When No Results Match

1. Relax soft constraints first
2. Expand query terms using ontology synonyms
3. Lower minimum confidence thresholds
4. Return partial matches with explanations

### When Multiple Intents Detected

1. Score for each intent separately
2. Combine results with intent-specific weights
3. Deduplicate by dataset_id, keeping highest score
4. Explain which intent drove each result

---

## Implementation Pseudocode

```python
def route_query(query: str) -> QueryIntent:
    """Determine query intent and build routing plan."""

    parsed = parse_query(query)

    # Check for explicit constraints
    has_exclusions = bool(parsed.excluded_modalities or parsed.excluded_species)
    has_dataset_ref = bool(re.search(r'DANDI|OpenNeuro|ds\d+', query))
    has_paper_ref = bool(re.search(r'paper|DOI|publication', query, re.I))
    has_task_terms = bool(parsed.tasks)
    has_modality_terms = bool(parsed.modalities)
    has_analysis_terms = bool(parsed.analysis)

    # Route based on strongest signal
    if has_dataset_ref and "similar" in query.lower():
        return QueryIntent.SIMILAR_DATASET_SEARCH

    if has_dataset_ref and has_paper_ref:
        if "paper" in query.lower():
            return QueryIntent.DATASET_TO_PAPER_LINKING
        return QueryIntent.PAPER_TO_DATASET_LINKING

    if has_dataset_ref:
        return QueryIntent.DATASET_LOOKUP

    if has_paper_ref:
        return QueryIntent.PAPER_LOOKUP

    if has_exclusions:
        return QueryIntent.NEGATIVE_CONSTRAINT_SEARCH

    if has_analysis_terms:
        return QueryIntent.ANALYSIS_AFFORDANCE_SEARCH

    if has_task_terms:
        return QueryIntent.TASK_SEARCH

    if has_modality_terms:
        return QueryIntent.MODALITY_REGION_SPECIES_SEARCH

    return QueryIntent.AMBIGUOUS_EXPLORATORY_SEARCH
```

---

## Integration with Existing Search

The query routing should:

1. Run **before** scoring to determine weights
2. Set `retrieval_config.weights` based on intent
3. Apply field-specific embedding queries
4. Preserve existing score decomposition
5. Add `query_intent` to response metadata

```python
def search_with_routing(query: str, ...) -> SearchResponse:
    intent = route_query(query)
    config = load_retrieval_config()

    # Override weights based on intent
    config["weights"] = INTENT_WEIGHTS[intent]

    # Run search with intent-specific config
    response = search_datasets(query, retrieval_config=config, ...)

    # Add routing metadata
    response.metadata["query_intent"] = intent.value
    response.metadata["applied_weights"] = config["weights"]

    return response
```
