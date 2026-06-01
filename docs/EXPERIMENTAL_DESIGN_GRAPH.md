# Experimental Design Graph

Experimental design seeds describe scientific data requirements that can be
matched against the knowledge graph.

Default seeds live at:

```text
data/graph/experimental_design_seeds.yaml
```

The loader accepts both mapping style:

```yaml
requires:
  tasks:
    - reversal_learning
```

and list-of-maps style:

```yaml
requires:
  - tasks: [reversal_learning]
```

## Matching

Use:

```python
from neural_search.graph import find_datasets_for_experimental_design

matches = find_datasets_for_experimental_design(
    graph,
    "q_learning_behavior_neural_experiment",
    min_score=0.5,
)
```

The matcher checks graph labels, analysis affordance nodes, and minimum dataset
usability flags stored on dataset-node properties.

## Caveats

Seeds are conservative planning aids. They should not create unsupported
scientific claims; every dataset match is still backed by graph edges and
evidence extracted from normalized records.
