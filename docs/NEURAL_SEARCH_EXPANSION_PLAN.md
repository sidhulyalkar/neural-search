# Neural Search v1.0 Expansion Plan

## Executive Summary

This document outlines the strategic roadmap for expanding Neural Search from a neuroscience dataset discovery tool into a comprehensive cross-disciplinary platform that bridges:

1. **Experimental Neuroscience** - Datasets from DANDI, OpenNeuro, Allen Brain, NeMO
2. **Computational Neuroscience** - Models from ModelDB, NeuroML, simulators
3. **Cognitive Neuroscience** - Behavioral paradigms, cognitive models
4. **Artificial Intelligence/ML** - Neural networks, training datasets, benchmarks
5. **Neuroinformatics** - Standards, formats, tools, analysis packages

The goal is to enable queries like:
- "Find datasets suitable for training RNN models of motor cortex dynamics"
- "Which computational models have been validated on Neuropixels data?"
- "Find deep networks that achieve high Brain-Score neural predictivity"

---

## Part 1: Corpus Expansion Strategy

### 1.1 Current State

**Covered Sources:**
- DANDI Archive (~500 datasets) - NWB format neurophysiology
- OpenNeuro (~3000 datasets) - BIDS format neuroimaging
- Allen Brain Observatory (~50 datasets) - Visual coding, connectivity

**Gaps:**
- Limited coverage of transcriptomics (NeMO)
- No computational model repositories
- No AI/ML paper/model integration
- Limited cross-repository linking

### 1.2 Phase 1: Neuroscience Dataset Repositories (Q1-Q2)

#### 1.2.1 NeMO Archive Integration

**Source:** https://nemoarchive.org/

**Data Types:**
- Single-cell RNA-seq
- Single-nucleus RNA-seq
- Spatial transcriptomics
- ATAC-seq chromatin accessibility

**Entity Schema:**
```yaml
entity_type: transcriptomic_dataset
fields:
  - dataset_id: nemo:{project_id}
  - species: [human, mouse, marmoset]
  - tissue_type: [brain_region, organoid, cell_line]
  - modality: [scRNA-seq, snRNA-seq, spatial, ATAC-seq]
  - cell_types: [list of annotated cell types]
  - reference_genome: [GRCh38, mm10]
  - analysis_affordances:
    - cell_type_clustering
    - differential_expression
    - trajectory_inference
    - gene_regulatory_networks
```

**New Edge Types:**
- `transcriptome_from_region`: Dataset → Brain Region
- `identifies_cell_type`: Dataset → Cell Type
- `expresses_gene`: Cell Type → Gene

#### 1.2.2 International Brain Laboratory (IBL)

**Source:** https://int-brain-lab.github.io/

**Value Proposition:**
- Standardized decision-making task across 10+ labs
- Neuropixels recordings from 100+ brain regions
- Gold standard for cross-lab reproducibility

**Entity Schema:**
```yaml
entity_type: ibl_dataset
fields:
  - dataset_id: ibl:{session_uuid}
  - lab: [list of IBL labs]
  - subject: {id, sex, strain, birth_date}
  - task_variant: ibl_choice_world
  - brain_regions: [computed from probe trajectories]
  - behavioral_events:
    - stimulus_onset
    - choice_time
    - feedback_time
    - wheel_movement
  - analysis_affordances:
    - choice_decoding
    - psychometric_modeling
    - drift_analysis
    - cross_lab_comparison
```

#### 1.2.3 Distributed Archives of Neurophysiology Data Integration Exchange (DANDI)

**Enhanced Coverage:**
- Implement incremental sync from DANDI API
- Extract NWB schema information for affordance detection
- Link to dandiset publications

**New Metadata Extraction:**
```python
# Enhanced NWB metadata extraction
class NWBAffordanceExtractor:
    """Extract analysis affordances from NWB file structure."""

    def extract(self, nwb_file):
        affordances = []

        # Check for spike trains
        if 'units' in nwb_file.processing:
            affordances.append('spike_analysis')
            if self._has_quality_metrics(nwb_file):
                affordances.append('population_dynamics')

        # Check for behavioral events
        if 'trials' in nwb_file.intervals:
            trials = nwb_file.intervals['trials']
            if 'choice' in trials.columns:
                affordances.append('choice_decoding')
            if 'reward' in trials.columns:
                affordances.append('reinforcement_learning')

        # Check for continuous signals
        if 'ecephys' in nwb_file.acquisition:
            affordances.append('lfp_analysis')
            affordances.append('spectral_analysis')

        return affordances
```

### 1.3 Phase 2: Computational Neuroscience Resources (Q2-Q3)

#### 1.3.1 ModelDB Integration

**Source:** https://modeldb.science/

**Entity Schema:**
```yaml
entity_type: computational_model
fields:
  - model_id: modeldb:{accession}
  - model_type: [compartmental, point_neuron, network, reduced]
  - neuron_model: [HH, IF, AdEx, conductance_based]
  - implemented_in: [NEURON, Brian, NEST, custom]
  - brain_region: [from model description]
  - phenomena_modeled:
    - action_potential
    - synaptic_plasticity
    - oscillations
    - decision_making
  - associated_paper: doi:{paper_doi}
  - simulation_requirements:
    - dt: [time step]
    - duration: [typical simulation time]
    - parameters: [list of key parameters]
```

**New Edge Types:**
- `model_of_region`: Model → Brain Region
- `validated_on_data`: Model → Dataset
- `implements_mechanism`: Model → Neural Mechanism
- `derived_from_model`: Model → Model

#### 1.3.2 Open Source Brain / NeuroML

**Source:** https://www.opensourcebrain.org/

**Value Proposition:**
- Standardized model descriptions in NeuroML
- Cross-simulator compatibility
- Version-controlled model evolution

**Integration Strategy:**
```python
class NeuroMLParser:
    """Parse NeuroML models for indexing."""

    def parse(self, neuroml_file):
        return {
            'model_id': f'osb:{self.extract_id(neuroml_file)}',
            'cell_types': self.extract_cell_types(neuroml_file),
            'channels': self.extract_ion_channels(neuroml_file),
            'synapses': self.extract_synapses(neuroml_file),
            'network_structure': self.extract_network(neuroml_file),
            'simulation_config': self.extract_config(neuroml_file),
        }
```

#### 1.3.3 Simulator Metadata

**Entity Schema:**
```yaml
entity_type: simulator
fields:
  - simulator_id: sim:{name}
  - name: [NEURON, NEST, Brian2, GeNN, Arbor]
  - model_types_supported: [point, compartmental, rate]
  - programming_interface: [Python, hoc, SLI]
  - gpu_support: boolean
  - parallel_support: [MPI, threads, GPU]
  - typical_use_cases:
    - single_neuron_simulation
    - network_simulation
    - large_scale_brain_model
```

### 1.4 Phase 3: AI/ML Resources (Q3-Q4)

#### 1.4.1 Papers With Code Integration

**Source:** https://paperswithcode.com/

**Entity Schema:**
```yaml
entity_type: ml_paper
fields:
  - paper_id: pwc:{paper_id}
  - title: string
  - arxiv_id: string
  - code_repos: [list of GitHub URLs]
  - methods_used: [attention, convolution, RNN, transformer]
  - datasets_used: [list of dataset names]
  - tasks_addressed: [classification, prediction, generation]
  - neuroscience_relevance:
    - brain_inspired: boolean
    - neural_data_analysis: boolean
    - cognitive_modeling: boolean
```

**Neuroscience-AI Linking:**
```python
# Keywords for identifying neuro-AI relevant papers
NEURO_AI_KEYWORDS = [
    'neural coding', 'neural network', 'brain-inspired',
    'biologically plausible', 'spiking neural', 'neural data',
    'neuroscience', 'brain activity', 'neural population',
    'cognitive model', 'decision making model', 'attention mechanism',
    'working memory', 'hippocampus', 'prefrontal cortex',
    'visual cortex', 'motor cortex', 'neural dynamics'
]

def is_neuro_ai_relevant(paper):
    """Check if ML paper is relevant to neuroscience."""
    text = (paper.title + ' ' + paper.abstract).lower()
    return any(kw in text for kw in NEURO_AI_KEYWORDS)
```

#### 1.4.2 Brain-Score Integration

**Source:** https://www.brain-score.org/

**Entity Schema:**
```yaml
entity_type: brain_score_benchmark
fields:
  - benchmark_id: brainscore:{name}
  - neural_region: [V1, V2, V4, IT, behavior]
  - metric_type: [neural_predictivity, behavioral_similarity]
  - stimuli: [ImageNet, COCO, gratings, naturalistic]
  - top_models: [list with scores]
```

**New Edge Types:**
- `achieves_score_on`: Model → Benchmark
- `predicts_neural_data`: Model → Dataset
- `inspired_by_region`: Model → Brain Region

#### 1.4.3 Hugging Face Model Hub

**Source:** https://huggingface.co/

**Filtering Strategy:**
- Focus on neuroscience-relevant models
- Include models trained on neural/brain data
- Include brain-inspired architectures

**Entity Schema:**
```yaml
entity_type: pretrained_model
fields:
  - model_id: hf:{org}/{name}
  - architecture: [transformer, CNN, RNN, hybrid]
  - task: [classification, generation, embedding]
  - training_data: [list of datasets]
  - neuroscience_relevance:
    - trained_on_brain_data: boolean
    - brain_inspired_architecture: boolean
    - neural_analysis_tool: boolean
```

### 1.5 Phase 4: Literature Integration (Q4+)

#### 1.5.1 OpenAlex Integration

**Source:** https://openalex.org/

**Value Proposition:**
- Open scholarly metadata for 200M+ works
- Citation graphs
- Author/institution linking
- Concept tagging

**Integration Strategy:**
```python
class OpenAlexLinker:
    """Link datasets to papers via OpenAlex."""

    def link_dataset_to_papers(self, dataset):
        # Search by DOI if available
        if dataset.publication_doi:
            paper = self.fetch_by_doi(dataset.publication_doi)
            yield ('linked_to_paper', paper)

        # Search by title similarity
        for paper in self.search_similar_titles(dataset.title):
            if self.is_likely_match(dataset, paper):
                yield ('related_paper', paper)

        # Extract citations for graph building
        for paper in dataset.linked_papers:
            for ref in self.get_references(paper):
                yield ('cites', ref)
```

#### 1.5.2 Semantic Scholar Integration

**Source:** https://www.semanticscholar.org/

**Value Proposition:**
- AI-extracted paper relationships
- TLDR summaries
- Influential citation detection

#### 1.5.3 bioRxiv/arXiv Preprints

**Focus Areas:**
- Neuroscience (q-bio.NC)
- Machine Learning (cs.LG, stat.ML)
- Computational Neuroscience (q-bio.NC)
- Neural and Evolutionary Computing (cs.NE)

---

## Part 2: Embedding Storage Architecture

### 2.1 Current State

**Current Approach:**
- JSONL files with pre-computed embeddings
- In-memory loading at startup
- No incremental updates

**Limitations:**
- Memory-bound scaling
- No efficient similarity search
- No embedding versioning

### 2.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Embedding Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────┐    ┌───────────┐    ┌───────────────────────┐  │
│   │  Source   │───▶│  Encoder  │───▶│   Vector Store        │  │
│   │  Entity   │    │  Models   │    │   (HNSW Index)        │  │
│   └───────────┘    └───────────┘    └───────────────────────┘  │
│                          │                      │               │
│                          ▼                      ▼               │
│                    ┌───────────┐    ┌───────────────────────┐  │
│                    │ Metadata  │    │   Provenance Store    │  │
│                    │   Store   │    │   (Version Control)   │  │
│                    └───────────┘    └───────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Multi-Space Embedding Design

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class EntityEmbedding:
    """Multi-space embedding for a single entity."""

    entity_id: str
    entity_type: str  # dataset, paper, model, etc.

    # Universal semantic space (for cross-type similarity)
    universal_embedding: np.ndarray  # shape: (768,)

    # Domain-specific embeddings
    domain_embeddings: Dict[str, np.ndarray]
    # e.g., {'task': (256,), 'modality': (256,), 'region': (256,)}

    # Provenance
    model_version: str
    computed_at: str
    source_hash: str  # Hash of source data for cache invalidation


@dataclass
class EmbeddingIndex:
    """HNSW index for efficient similarity search."""

    index_type: str  # 'universal', 'task', 'modality', etc.
    dimension: int
    num_entities: int
    index_params: Dict  # M, ef_construction, etc.

    def search(self, query_vector: np.ndarray, k: int) -> List[str]:
        """Return top-k entity IDs."""
        pass

    def add(self, entity_id: str, vector: np.ndarray):
        """Add or update entity in index."""
        pass
```

### 2.4 Storage Backend Options

#### Option A: Local File-Based (Current + Enhanced)

```python
class LocalEmbeddingStore:
    """Enhanced local storage with HNSW indexing."""

    def __init__(self, base_path: str):
        self.vectors_path = base_path / "vectors"
        self.index_path = base_path / "indices"
        self.metadata_path = base_path / "metadata"

    def build_index(self, embedding_type: str):
        """Build HNSW index for embedding type."""
        import hnswlib

        vectors = self.load_vectors(embedding_type)
        index = hnswlib.Index(space='cosine', dim=vectors.shape[1])
        index.init_index(max_elements=len(vectors) * 2, M=16, ef_construction=200)
        index.add_items(vectors, range(len(vectors)))
        index.save_index(self.index_path / f"{embedding_type}.bin")
```

#### Option B: Vector Database (Scaling)

```python
class VectorDBStore:
    """Vector database backend for larger scale."""

    def __init__(self, connection_string: str):
        # Options: Milvus, Pinecone, Weaviate, Qdrant
        self.client = self._connect(connection_string)

    def create_collection(self, name: str, schema: EmbeddingSchema):
        """Create collection with schema."""
        pass

    def upsert(self, entities: List[EntityEmbedding]):
        """Insert or update embeddings."""
        pass

    def search(self, query: np.ndarray, filter: Dict, k: int):
        """Filtered similarity search."""
        pass
```

### 2.5 Embedding Versioning

```python
@dataclass
class EmbeddingVersion:
    """Track embedding model versions and data state."""

    version_id: str  # e.g., "v1.2.3"
    model_name: str  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    model_hash: str  # Hash of model weights
    training_data_hash: str  # Hash of fine-tuning data
    created_at: str

    # Compatibility
    supersedes: Optional[str]  # Previous version
    breaking_changes: bool  # Requires reindex?


class EmbeddingVersionManager:
    """Manage embedding version migrations."""

    def needs_recompute(self, entity_id: str, current_version: str) -> bool:
        """Check if entity needs embedding update."""
        stored_version = self.get_entity_version(entity_id)
        return stored_version != current_version

    def migrate(self, from_version: str, to_version: str):
        """Migrate embeddings between versions."""
        entities_to_update = self.find_outdated(from_version)
        for batch in self.batch_entities(entities_to_update):
            new_embeddings = self.encoder.encode_batch(batch)
            self.store.upsert(new_embeddings, version=to_version)
```

---

## Part 3: Cross-Domain Knowledge Graph Extension

### 3.1 Extended Node Types

```yaml
# Experimental Neuroscience (existing)
node_types:
  - dataset: Experimental dataset
  - paper: Scientific publication
  - task: Behavioral task/paradigm
  - modality: Recording modality
  - species: Organism
  - region: Brain region
  - event: Behavioral event
  - analysis: Analysis method

# Computational Neuroscience (new)
  - computational_model: Simulated model
  - neural_mechanism: Biological mechanism
  - ion_channel: Ion channel model
  - synapse_model: Synaptic mechanism
  - simulator: Simulation software

# AI/ML (new)
  - ml_model: Machine learning model
  - architecture: Network architecture type
  - benchmark: Evaluation benchmark
  - training_dataset: ML training data
  - framework: ML framework (PyTorch, TF)

# Cross-domain (new)
  - concept: Scientific concept
  - method: Methodology/technique
  - software: Analysis software
  - data_format: File format/standard
```

### 3.2 Extended Edge Types

```yaml
# Dataset relationships (existing)
edge_types:
  - has_task: Dataset → Task
  - has_modality: Dataset → Modality
  - has_species: Dataset → Species
  - records_region: Dataset → Region
  - linked_to_paper: Dataset → Paper

# Model relationships (new)
  - model_of_region: ComputationalModel → Region
  - model_of_mechanism: ComputationalModel → NeuralMechanism
  - validated_on: ComputationalModel → Dataset
  - implemented_in: ComputationalModel → Simulator
  - derived_from: ComputationalModel → ComputationalModel

# AI-Neuro relationships (new)
  - trained_on: MLModel → Dataset
  - achieves_on: MLModel → Benchmark
  - inspired_by: MLModel → NeuralMechanism
  - predicts: MLModel → Dataset
  - architecture_from: MLModel → Architecture

# Literature relationships (new)
  - introduces: Paper → Method
  - validates: Paper → ComputationalModel
  - analyzes: Paper → Dataset
  - cites: Paper → Paper
  - reviews: Paper → Concept

# Cross-domain (new)
  - implements: Software → Method
  - supports_format: Software → DataFormat
  - used_by: Method → Analysis
  - compatible_with: Dataset → DataFormat
```

### 3.3 Cross-Domain Metapath Templates

```python
CROSS_DOMAIN_METAPATHS = {
    # Find datasets suitable for training specific model types
    'dataset_for_model_training': [
        ('Dataset', 'has_task', 'Task'),
        ('Task', 'modeled_by', 'ComputationalModel'),
        ('ComputationalModel', 'architecture_type', 'Architecture'),
    ],

    # Find models validated on similar data
    'model_validated_on_similar': [
        ('Dataset', 'similar_to', 'Dataset'),
        ('Dataset', 'validates', 'ComputationalModel'),
    ],

    # Find papers linking experiments to models
    'experiment_to_model_paper': [
        ('Dataset', 'linked_to_paper', 'Paper'),
        ('Paper', 'validates', 'ComputationalModel'),
    ],

    # Find ML models inspired by brain region
    'brain_inspired_ml': [
        ('Region', 'studied_in', 'Dataset'),
        ('Dataset', 'linked_to_paper', 'Paper'),
        ('Paper', 'inspired', 'MLModel'),
    ],

    # Find analysis methods used on similar data
    'analysis_transfer': [
        ('Dataset', 'similar_to', 'Dataset'),
        ('Dataset', 'analyzed_with', 'Method'),
        ('Method', 'implemented_in', 'Software'),
    ],
}
```

---

## Part 4: Implementation Roadmap

### 4.1 Phase 1: Foundation (Months 1-2)

**Goals:**
- Implement enhanced embedding storage
- Add HNSW indexing
- Implement version management

**Deliverables:**
1. `neural_search/embeddings/storage.py` - New storage backend
2. `neural_search/embeddings/indexing.py` - HNSW integration
3. `neural_search/embeddings/versioning.py` - Version management
4. Tests for all new modules

### 4.2 Phase 2: Corpus Expansion (Months 2-4)

**Goals:**
- Integrate NeMO Archive
- Integrate IBL data
- Enhance DANDI metadata extraction

**Deliverables:**
1. `neural_search/ingestion/nemo.py` - NeMO ingestion
2. `neural_search/ingestion/ibl.py` - IBL ingestion
3. Enhanced `neural_search/ingestion/dandi.py`
4. New ontology entries for transcriptomics

### 4.3 Phase 3: Computational Models (Months 4-6)

**Goals:**
- Integrate ModelDB
- Integrate Open Source Brain
- Build model-dataset linking

**Deliverables:**
1. `neural_search/ingestion/modeldb.py`
2. `neural_search/ingestion/osb.py`
3. `neural_search/graph/model_edges.py`
4. New metapath templates

### 4.4 Phase 4: AI/ML Integration (Months 6-8)

**Goals:**
- Integrate Papers With Code (filtered)
- Integrate Brain-Score
- Build neuro-AI concept linking

**Deliverables:**
1. `neural_search/ingestion/paperswithcode.py`
2. `neural_search/ingestion/brainscore.py`
3. `neural_search/ontology/neuro_ai_concepts.py`
4. Cross-domain search features

### 4.5 Phase 5: Literature Graph (Months 8-10)

**Goals:**
- Integrate OpenAlex
- Build citation graph
- Enable literature-based discovery

**Deliverables:**
1. `neural_search/ingestion/openalex.py`
2. `neural_search/graph/citation_edges.py`
3. Literature-based search features

---

## Part 5: Success Metrics

### 5.1 Coverage Metrics

| Metric | Current | Target (6mo) | Target (12mo) |
|--------|---------|--------------|---------------|
| Total datasets | 3,500 | 8,000 | 15,000 |
| Computational models | 0 | 500 | 2,000 |
| ML papers (neuro-relevant) | 0 | 1,000 | 5,000 |
| Cross-domain edges | 0 | 5,000 | 25,000 |
| Sources integrated | 3 | 8 | 15 |

### 5.2 Quality Metrics

| Metric | Current | Target |
|--------|---------|--------|
| P@5 (benchmark) | 76.7% | 80%+ |
| MRR | 0.950 | 0.96+ |
| Hard-negative violations | 0 | 0 |
| Cross-domain query success | N/A | 70%+ |

### 5.3 Usage Metrics

| Metric | Target |
|--------|--------|
| Queries per month | 1,000+ |
| Unique users | 100+ |
| Cross-domain queries | 20%+ |
| API integrations | 5+ |

---

## Appendix A: Example Cross-Domain Queries

### Query 1: Model Training Data
```
"Find mouse visual cortex datasets with trial-aligned spike trains
suitable for training RNN models of visual processing"
```

**Expected Behavior:**
1. Filter: species=mouse, region=visual_cortex, modality=electrophysiology
2. Affordance check: has_spike_trains, has_trial_alignment
3. Cross-domain: link to RNN model papers trained on similar data

### Query 2: Model Validation
```
"Which computational models of decision-making have been validated
on Neuropixels recordings from prefrontal cortex?"
```

**Expected Behavior:**
1. Entity search: ComputationalModel with task=decision_making
2. Graph traversal: validated_on edges to Dataset
3. Filter: Dataset with modality=neuropixels, region=PFC

### Query 3: Brain-Inspired AI
```
"Find transformer models that incorporate biologically plausible
attention mechanisms inspired by visual cortex"
```

**Expected Behavior:**
1. Entity search: MLModel with architecture=transformer
2. Concept match: biologically_plausible_attention
3. Graph traversal: inspired_by → visual_cortex mechanisms

---

## Appendix B: Data Format Specifications

### B.1 Extended Dataset Schema

```python
class ExtendedDatasetRecord(BaseModel):
    """Extended schema for cross-domain datasets."""

    # Core fields (existing)
    dataset_id: str
    source: str
    source_id: str
    title: str
    description: str

    # Experimental context (existing)
    species: List[str]
    tasks: List[str]
    modalities: List[str]
    brain_regions: List[str]

    # Extended fields (new)
    data_type: Literal['experimental', 'simulated', 'derived']
    file_formats: List[str]  # NWB, BIDS, HDF5, etc.

    # Model compatibility (new)
    compatible_simulators: List[str]
    compatible_frameworks: List[str]  # PyTorch, TF, etc.

    # ML relevance (new)
    ml_task_types: List[str]  # classification, regression, etc.
    suggested_architectures: List[str]

    # Provenance (enhanced)
    extraction_methods: List[ExtractionMethod]
    linked_models: List[str]  # IDs of validated models
    linked_papers: List[str]  # DOIs
```

### B.2 Computational Model Schema

```python
class ComputationalModelRecord(BaseModel):
    """Schema for computational neuroscience models."""

    model_id: str
    source: str  # modeldb, osb, github
    source_id: str

    # Model type
    model_type: Literal['compartmental', 'point_neuron',
                        'rate', 'spiking_network', 'cognitive']

    # Implementation
    implemented_in: List[str]  # NEURON, Brian2, etc.
    code_repository: Optional[str]

    # Scientific content
    modeled_regions: List[str]
    modeled_mechanisms: List[str]
    phenomena_reproduced: List[str]

    # Validation
    validated_on_datasets: List[str]
    validation_metrics: Dict[str, float]

    # Associated literature
    original_paper: str  # DOI
    citing_papers: List[str]
```

---

*Document Version: 1.0*
*Last Updated: May 2026*
