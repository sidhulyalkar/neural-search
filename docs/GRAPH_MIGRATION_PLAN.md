# Graph Database Migration Plan

This document outlines the plan to migrate the Neural Search knowledge graph from file-based JSON storage to a proper graph database.

## Current State

### File-Based Storage
- **Format**: JSON/JSONL files
- **Location**: `data/graph/neural_search_graph.*.json`
- **Size**: ~146 nodes, ~357 edges (current corpus)
- **Scale limit**: ~5,000 datasets before performance degrades

### Current Capabilities
- 39 node types (Dataset, Paper, Task, Modality, etc.)
- 39 edge types (dataset_has_modality, paper_uses_dataset, etc.)
- BFS transitive expansion (max 2 hops)
- Provenance-aware edges with confidence scores

### Current Limitations
1. Full graph loaded into memory
2. No real-time updates (requires reindexing)
3. No ACID transactions
4. Limited query expressiveness
5. Scaling ceiling around 5-10k nodes

## Migration Goals

### Short-term (3-6 months)
1. Seamless migration path (no data loss)
2. Maintain backward compatibility with existing API
3. Enable real-time graph updates
4. Support concurrent read/write access

### Medium-term (6-12 months)
1. Scale to 50,000+ datasets
2. Advanced graph queries (path finding, community detection)
3. Graph-based ML features (node embeddings, link prediction)
4. Versioned graph with temporal queries

## Database Evaluation

### Option 1: Neo4j (Recommended)
**Pros:**
- Mature, well-documented
- Cypher query language is expressive
- Excellent Python support (neo4j-driver)
- Graph data science library for ML
- Community edition is free

**Cons:**
- Requires separate server
- Memory-intensive for large graphs
- Enterprise features require license

**Estimated effort:** 3-4 weeks

### Option 2: Amazon Neptune
**Pros:**
- Managed service (no ops overhead)
- Scales automatically
- Gremlin and SPARQL support

**Cons:**
- AWS lock-in
- Cost at scale
- Less expressive than Cypher

**Estimated effort:** 4-5 weeks

### Option 3: SQLite + Extension (DuckDB Graph)
**Pros:**
- Zero-dependency option
- Good for single-user scenarios
- Can use existing SQLite infrastructure

**Cons:**
- Limited graph-native features
- Less mature for graph workloads
- May not scale well

**Estimated effort:** 2-3 weeks

### Option 4: NetworkX + Persistent Backend
**Pros:**
- Already using NetworkX for analysis
- Can add persistence layer
- No new dependencies

**Cons:**
- Still in-memory constrained
- No transaction support
- Custom implementation required

**Estimated effort:** 2-3 weeks

## Recommended Approach: Neo4j

### Phase 1: Schema Design (Week 1)

#### Node Labels
```cypher
// Core entities
(:Dataset {id, title, source, source_id, description})
(:Paper {id, title, doi, abstract})
(:Task {id, label, category})
(:Modality {id, label, category})
(:Species {id, label, taxon_group})
(:BrainRegion {id, label, hierarchy_path})
(:AnalysisAffordance {id, label, requirements})

// Supporting entities
(:Author {id, name, orcid})
(:DataStandard {id, label, version})
(:RecordingDevice {id, label, manufacturer})
```

#### Relationship Types
```cypher
// Dataset relationships
(d:Dataset)-[:HAS_MODALITY {confidence, source}]->(m:Modality)
(d:Dataset)-[:HAS_TASK {confidence, source}]->(t:Task)
(d:Dataset)-[:HAS_SPECIES {confidence, source}]->(s:Species)
(d:Dataset)-[:RECORDS_FROM {confidence}]->(r:BrainRegion)
(d:Dataset)-[:SUPPORTS_ANALYSIS {confidence, requirements_met}]->(a:AnalysisAffordance)

// Paper relationships
(p:Paper)-[:USES_DATASET {explicit, confidence}]->(d:Dataset)
(p:Paper)-[:MENTIONS_DATASET {confidence}]->(d:Dataset)
(p:Paper)-[:AUTHORED_BY {position}]->(a:Author)

// Analysis requirements
(a:AnalysisAffordance)-[:REQUIRES_MODALITY]->(m:Modality)
(a:AnalysisAffordance)-[:REQUIRES_BEHAVIORAL_EVENT]->(e:BehavioralEvent)
```

### Phase 2: Migration Script (Week 2)

```python
# migration/json_to_neo4j.py

from neo4j import GraphDatabase
from neural_search.graph.schema import load_graph

def migrate_to_neo4j(json_path: str, neo4j_uri: str, auth: tuple):
    """Migrate JSON graph to Neo4j."""

    # Load existing graph
    graph = load_graph(json_path)

    # Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=auth)

    with driver.session() as session:
        # Create constraints
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE d.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")

        # Migrate nodes
        for node in graph.nodes:
            node_data = graph.nodes[node]
            label = node_data.get("node_type", "Unknown")
            session.run(
                f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                id=node,
                props=node_data
            )

        # Migrate edges
        for source, target, data in graph.edges(data=True):
            edge_type = data.get("edge_type", "RELATED_TO")
            session.run(
                f"""
                MATCH (a {{id: $source}}), (b {{id: $target}})
                MERGE (a)-[r:{edge_type}]->(b)
                SET r += $props
                """,
                source=source,
                target=target,
                props=data
            )

    driver.close()
```

### Phase 3: Query Adapter (Week 3)

```python
# neural_search/graph/neo4j_backend.py

class Neo4jGraphBackend:
    """Neo4j backend for graph queries."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_dataset_context(self, dataset_id: str) -> dict:
        """Get graph context for a dataset (replaces graph_context_score)."""
        query = """
        MATCH (d:Dataset {id: $id})
        OPTIONAL MATCH (d)-[:HAS_MODALITY]->(m:Modality)
        OPTIONAL MATCH (d)-[:HAS_TASK]->(t:Task)
        OPTIONAL MATCH (d)-[:SUPPORTS_ANALYSIS]->(a:AnalysisAffordance)
        OPTIONAL MATCH (p:Paper)-[:USES_DATASET]->(d)
        RETURN d, collect(DISTINCT m) as modalities,
               collect(DISTINCT t) as tasks,
               collect(DISTINCT a) as affordances,
               collect(DISTINCT p) as papers
        """
        with self.driver.session() as session:
            result = session.run(query, id=dataset_id)
            return self._format_context(result.single())

    def expand_query_with_graph(self, query_terms: dict, max_hops: int = 2) -> dict:
        """Expand query using graph relationships."""
        query = """
        UNWIND $task_ids as task_id
        MATCH (t:Task {id: task_id})
        OPTIONAL MATCH (t)-[:SIMILAR_TO*1..{max_hops}]-(related:Task)
        RETURN t, collect(DISTINCT related) as related_tasks
        """
        # ... implementation

    def find_similar_datasets(self, dataset_id: str, limit: int = 10) -> list:
        """Find datasets similar via graph structure."""
        query = """
        MATCH (d:Dataset {id: $id})
        MATCH (d)-[:HAS_MODALITY]->(m)<-[:HAS_MODALITY]-(other:Dataset)
        WHERE other.id <> $id
        WITH other, count(m) as shared_modalities
        MATCH (d)-[:HAS_TASK]->(t)<-[:HAS_TASK]-(other)
        WITH other, shared_modalities, count(t) as shared_tasks
        RETURN other.id as id,
               shared_modalities + shared_tasks as similarity_score
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, id=dataset_id, limit=limit)
            return [dict(r) for r in result]
```

### Phase 4: Integration (Week 4)

1. Add Neo4j backend as optional provider
2. Update `search_features.py` to use Neo4j when configured
3. Add configuration for Neo4j connection
4. Add tests for Neo4j backend
5. Document migration process

### Configuration

```yaml
# data/config/graph.yaml

graph:
  backend: neo4j  # or "json" for backward compatibility

  neo4j:
    uri: bolt://localhost:7687
    user: neo4j
    password: ${NEO4J_PASSWORD}
    database: neuralsearch

  json:
    path: data/graph/neural_search_graph.real_corpus.json

  features:
    max_transitive_hops: 2
    use_edge_confidence: true
```

## Migration Checklist

### Pre-Migration
- [ ] Export current graph to backup
- [ ] Set up Neo4j instance (local or cloud)
- [ ] Run migration script on test data
- [ ] Verify node/edge counts match
- [ ] Test query performance

### Migration
- [ ] Schedule maintenance window
- [ ] Run full migration
- [ ] Verify data integrity
- [ ] Update configuration
- [ ] Run integration tests

### Post-Migration
- [ ] Monitor query latency
- [ ] Check memory usage
- [ ] Validate search quality
- [ ] Document any issues
- [ ] Remove deprecated JSON code (after validation period)

## Rollback Plan

1. Keep JSON files for 3 months post-migration
2. Add feature flag to switch backends
3. Maintain parity tests between backends
4. Document rollback procedure

## Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Evaluation | 1 week | Decision on database |
| Schema Design | 1 week | Schema documented |
| Migration Script | 1 week | Script tested |
| Query Adapter | 1 week | Adapter complete |
| Integration | 1 week | Tests passing |
| Testing | 1 week | QA complete |
| Deployment | 1 week | Production migration |

**Total estimated time:** 6-8 weeks

## Future Enhancements (Post-Migration)

1. **Graph ML Integration**
   - Node2Vec embeddings for similarity
   - Link prediction for paper-dataset linking
   - Community detection for dataset clustering

2. **Temporal Queries**
   - Version history for nodes/edges
   - "Graph at time T" queries
   - Change tracking

3. **Advanced Graph Features**
   - Shortest path queries
   - Subgraph extraction
   - Graph pattern matching
