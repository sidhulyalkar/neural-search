# AI Research Workflow Roadmap

This document defines the agent architecture for Neural Search's AI-assisted research workflows. Each agent type serves a specific role in the research lifecycle, from dataset discovery to experimental design to notebook generation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neural Search Agent System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  Dataset         │    │  Paper-to-Dataset │                  │
│  │  Discovery       │───▶│  Linking          │                  │
│  │  Agent           │    │  Agent            │                  │
│  └──────────────────┘    └──────────────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌──────────────────────────────────────────┐                   │
│  │           Corpus Knowledge Base           │                   │
│  │  (Datasets + Papers + Ontology + Graph)   │                   │
│  └──────────────────────────────────────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  Experimental    │    │  Benchmark/Audit  │                  │
│  │  Design          │    │  Agent            │                  │
│  │  Agent           │    │                   │                  │
│  └──────────────────┘    └──────────────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌──────────────────────────────────────────┐                   │
│  │         Notebook Generation Agent         │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Dataset Discovery Agent

### Purpose

Find relevant datasets for a research question by interpreting natural language queries, matching against the ontology, and ranking results by scientific relevance and analysis readiness.

### Capabilities

```yaml
dataset_discovery_agent:
  name: "DatasetDiscoveryAgent"
  description: >
    Interprets researcher queries and finds matching datasets
    from the Neural Search corpus with explanations.

  capabilities:
    # Query interpretation
    - parse_natural_language_query
    - extract_structured_constraints
    - map_to_ontology_terms
    - identify_implicit_requirements

    # Search execution
    - hybrid_retrieval  # text + ontology + metadata
    - constraint_filtering
    - analysis_affordance_matching
    - readiness_scoring

    # Result explanation
    - generate_match_evidence
    - highlight_partial_matches
    - explain_ranking_factors
    - suggest_query_refinements

  inputs:
    - natural_language_query: str
    - constraints: Optional[Dict]  # explicit filters
    - context: Optional[str]  # research context
    - num_results: int = 10

  outputs:
    - ranked_datasets: List[DatasetResult]
    - query_interpretation: QueryParsing
    - suggested_refinements: List[str]
```

### Implementation Specification

```python
class DatasetDiscoveryAgent:
    """
    Agent for finding relevant datasets given research queries.
    """

    def __init__(self, corpus, ontology, embeddings):
        self.corpus = corpus
        self.ontology = ontology
        self.embeddings = embeddings

    def discover(
        self,
        query: str,
        constraints: dict = None,
        context: str = None,
        num_results: int = 10,
    ) -> DiscoveryResult:
        """
        Main discovery workflow.

        Steps:
        1. Parse query into structured form
        2. Extract ontology matches
        3. Apply constraints
        4. Execute hybrid retrieval
        5. Score and rank results
        6. Generate explanations
        """

        # Step 1: Query parsing
        parsed = self.parse_query(query, context)
        # Output: {
        #   "task_intent": ["reversal_learning", "go_nogo"],
        #   "modality_intent": ["fiber_photometry"],
        #   "region_intent": ["VTA", "striatum"],
        #   "species_intent": ["mouse"],
        #   "analysis_intent": ["reward_prediction_error"],
        #   "free_text": "dopamine recordings during learning"
        # }

        # Step 2: Ontology matching
        ontology_matches = self.ontology.match(parsed)
        # Expands synonyms, finds related terms

        # Step 3: Constraint application
        effective_constraints = self.merge_constraints(
            parsed_constraints=parsed,
            explicit_constraints=constraints,
        )

        # Step 4: Hybrid retrieval
        candidates = self.retrieval.search(
            text_query=parsed["free_text"],
            ontology_terms=ontology_matches,
            constraints=effective_constraints,
            limit=num_results * 3,  # over-retrieve for reranking
        )

        # Step 5: Scoring and ranking
        scored = self.score_results(candidates, parsed)

        # Step 6: Generate explanations
        results = []
        for dataset, score in scored[:num_results]:
            explanation = self.explain_match(dataset, parsed, score)
            results.append(DatasetResult(
                dataset=dataset,
                score=score,
                explanation=explanation,
            ))

        return DiscoveryResult(
            results=results,
            query_parsing=parsed,
            refinements=self.suggest_refinements(parsed, scored),
        )

    def parse_query(self, query: str, context: str = None) -> dict:
        """
        Parse natural language query into structured intent.

        Uses:
        - Named entity recognition for scientific terms
        - Ontology lookup for task/behavior/region matching
        - Context extraction for implicit requirements
        """
        # Implementation: LLM-based parsing with ontology grounding
        pass

    def score_results(self, candidates, parsed) -> List[Tuple]:
        """
        Multi-factor scoring combining:
        - Text similarity (semantic)
        - Ontology match coverage
        - Constraint satisfaction
        - Analysis affordance fit
        - Data quality/readiness
        """
        pass

    def explain_match(self, dataset, parsed, score) -> MatchExplanation:
        """
        Generate human-readable explanation of why this dataset matches.

        Includes:
        - Matched ontology terms
        - Satisfied constraints
        - Analysis affordances available
        - Potential limitations or gaps
        """
        pass
```

### Example Interactions

```yaml
example_queries:
  - query: "dopamine recordings during reward learning in mice"
    interpretation:
      task_intent: [reversal_learning, classical_conditioning, bandit_task]
      modality_intent: [fiber_photometry, extracellular_ephys]
      region_intent: [VTA, SNc, NAc, striatum]
      species_intent: [mouse]
      behavior_intent: [reward, reward_prediction_error]
    top_result_explanation: >
      Dataset matches: fiber photometry in VTA during reversal learning
      in mice. Contains reward timestamps and choice data. Supports
      Q-learning model fitting and TD error analysis.

  - query: "human speech production ECoG for BCI"
    interpretation:
      task_intent: [speech_production, speech_bci]
      modality_intent: [ecog]
      region_intent: [speech_motor_cortex, Broca_area, ventral_sensorimotor]
      species_intent: [human]
      analysis_intent: [speech_decoding, phoneme_decoding]
    top_result_explanation: >
      Dataset matches: ECoG recordings during overt speech production
      in epilepsy patients. Contains phoneme-aligned neural data.
      Supports speech decoding analysis with high channel count.
```

---

## 2. Paper-to-Dataset Linking Agent

### Purpose

Establish bidirectional links between scientific literature and datasets. Find papers that describe or use specific datasets, and find datasets associated with papers.

### Capabilities

```yaml
paper_dataset_linking_agent:
  name: "PaperDatasetLinkingAgent"
  description: >
    Links papers to datasets and datasets to papers through
    multiple evidence sources.

  capabilities:
    # Paper → Dataset
    - extract_data_availability_statements
    - parse_repository_urls
    - match_dataset_names_fuzzy
    - identify_dataset_dois

    # Dataset → Paper
    - find_primary_publication
    - find_citing_papers
    - find_methods_papers
    - find_reuse_papers

    # Link validation
    - verify_link_evidence
    - compute_link_confidence
    - resolve_ambiguous_matches

    # Cross-reference
    - find_similar_datasets_via_papers
    - find_similar_papers_via_datasets
    - build_citation_network_links

  inputs:
    - paper_doi: Optional[str]
    - paper_title: Optional[str]
    - dataset_id: Optional[str]
    - link_direction: Literal["paper_to_data", "data_to_paper", "both"]

  outputs:
    - links: List[DatasetPaperLink]
    - confidence_scores: Dict[str, float]
    - evidence: Dict[str, List[str]]
```

### Implementation Specification

```python
class PaperDatasetLinkingAgent:
    """
    Agent for establishing paper-dataset links.
    """

    def __init__(self, corpus, openalex_client, crossref_client):
        self.corpus = corpus
        self.openalex = openalex_client
        self.crossref = crossref_client

    def link_paper_to_datasets(self, paper_doi: str) -> List[DatasetLink]:
        """
        Find datasets associated with a paper.

        Evidence sources:
        1. Data availability statement in paper
        2. Supplementary materials
        3. Repository links in paper text
        4. Author-deposited datasets
        5. Citation network (papers citing data papers)
        """

        paper = self.fetch_paper(paper_doi)

        links = []

        # Method 1: Parse data availability statement
        if paper.data_availability:
            das_links = self.parse_data_availability(paper.data_availability)
            links.extend(das_links)

        # Method 2: Check known repositories
        repo_links = self.search_repositories(
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
        )
        links.extend(repo_links)

        # Method 3: Author lookup
        author_datasets = self.find_author_datasets(
            authors=paper.authors,
            topic_filter=paper.topics,
        )
        links.extend(author_datasets)

        # Method 4: Citation network
        cited_data_papers = self.find_data_paper_citations(paper)
        links.extend(cited_data_papers)

        # Deduplicate and score
        unique_links = self.deduplicate_links(links)
        scored_links = self.score_links(unique_links, paper)

        return scored_links

    def link_dataset_to_papers(self, dataset_id: str) -> List[PaperLink]:
        """
        Find papers associated with a dataset.

        Evidence sources:
        1. Dataset metadata (related publications)
        2. OpenAlex search for dataset mention
        3. Citation search for data paper
        4. Author publication search
        """

        dataset = self.corpus.get(dataset_id)

        links = []

        # Method 1: Metadata links
        if dataset.related_papers:
            for paper_info in dataset.related_papers:
                links.append(PaperLink(
                    doi=paper_info.doi,
                    relation="metadata_linked",
                    confidence=0.95,
                ))

        # Method 2: OpenAlex search
        search_results = self.openalex.search(
            query=f'"{dataset.name}" OR "{dataset_id}"',
            filter_concepts=dataset.topics,
        )
        links.extend(self.process_search_results(search_results))

        # Method 3: Author search
        if dataset.contributors:
            author_papers = self.find_author_papers(
                authors=dataset.contributors,
                topic_filter=dataset.topics,
                year_range=(dataset.year - 2, dataset.year + 3),
            )
            links.extend(author_papers)

        # Score and rank
        return self.rank_paper_links(links, dataset)

    def parse_data_availability(self, text: str) -> List[DatasetLink]:
        """
        Extract dataset references from data availability statement.

        Patterns matched:
        - Repository URLs (DANDI, OpenNeuro, Figshare, Zenodo, OSF)
        - DOIs
        - Accession numbers
        - Dataset names
        """
        patterns = {
            "dandi": r"DANDI[:\s]*(\d{6})",
            "openneuro": r"(?:OpenNeuro|ds)(\d{6})",
            "zenodo": r"zenodo\.org/record/(\d+)",
            "figshare": r"figshare\.com/articles/(\d+)",
            "doi": r"10\.\d{4,}/[^\s]+",
            "osf": r"osf\.io/([a-z0-9]+)",
        }
        # Implementation: regex + validation
        pass

    def score_links(self, links, reference) -> List[ScoredLink]:
        """
        Score link confidence based on evidence strength.

        High confidence (>0.9):
        - Explicit DOI in data availability
        - Dataset metadata lists paper

        Medium confidence (0.6-0.9):
        - Author overlap + topic match + temporal proximity
        - Dataset name mentioned in paper

        Low confidence (0.3-0.6):
        - Topic match only
        - Citation network inference
        """
        pass
```

### Link Evidence Schema

```yaml
link_evidence_types:
  - type: "explicit_doi"
    description: "Paper contains dataset DOI"
    confidence: 0.95

  - type: "explicit_accession"
    description: "Paper contains dataset accession number"
    confidence: 0.95

  - type: "data_availability_url"
    description: "Data availability statement links to dataset"
    confidence: 0.90

  - type: "dataset_metadata"
    description: "Dataset metadata lists paper DOI"
    confidence: 0.95

  - type: "author_match"
    description: "Overlapping authors with topic/temporal match"
    confidence: 0.70

  - type: "name_mention"
    description: "Dataset name fuzzy-matched in paper text"
    confidence: 0.60

  - type: "citation_network"
    description: "Inferred through citation relationships"
    confidence: 0.50

  - type: "topic_match"
    description: "Same scientific topic without direct evidence"
    confidence: 0.30
```

---

## 3. Experimental Design Agent

### Purpose

Help researchers design experiments by providing reference designs, standard parameters, and example datasets for specific paradigms.

### Capabilities

```yaml
experimental_design_agent:
  name: "ExperimentalDesignAgent"
  description: >
    Assists with experimental design by retrieving reference
    implementations and standard parameters from the corpus.

  capabilities:
    # Design retrieval
    - find_reference_designs
    - extract_task_parameters
    - compare_design_variants
    - identify_best_practices

    # Parameter recommendations
    - recommend_timing_parameters
    - recommend_trial_structure
    - recommend_training_protocols
    - recommend_analysis_pipelines

    # Recording setup
    - recommend_recording_targets
    - recommend_electrode_configurations
    - recommend_imaging_parameters
    - recommend_behavioral_measures

    # Validation
    - check_design_completeness
    - identify_potential_confounds
    - suggest_control_conditions

  inputs:
    - task_type: str
    - species: str
    - modality: str
    - research_question: Optional[str]
    - constraints: Optional[Dict]

  outputs:
    - reference_designs: List[DesignReference]
    - parameter_recommendations: ParameterSet
    - example_datasets: List[Dataset]
    - design_checklist: List[str]
```

### Implementation Specification

```python
class ExperimentalDesignAgent:
    """
    Agent for experimental design assistance.
    """

    def __init__(self, corpus, ontology, design_templates):
        self.corpus = corpus
        self.ontology = ontology
        self.templates = design_templates

    def get_design_recommendations(
        self,
        task_type: str,
        species: str,
        modality: str,
        research_question: str = None,
    ) -> DesignRecommendations:
        """
        Generate experimental design recommendations.

        Steps:
        1. Find matching datasets in corpus
        2. Extract design parameters from datasets
        3. Aggregate to find common patterns
        4. Generate recommendations with evidence
        """

        # Step 1: Find reference datasets
        reference_datasets = self.corpus.search(
            task=task_type,
            species=species,
            modality=modality,
            min_quality=7,  # only high-quality references
        )

        # Step 2: Extract parameters
        design_params = []
        for dataset in reference_datasets:
            params = self.extract_design_parameters(dataset)
            design_params.append(params)

        # Step 3: Aggregate patterns
        common_patterns = self.aggregate_parameters(design_params)

        # Step 4: Generate recommendations
        recommendations = DesignRecommendations(
            task_structure=self.recommend_task_structure(
                task_type, common_patterns
            ),
            timing_parameters=self.recommend_timing(common_patterns),
            trial_counts=self.recommend_trial_counts(common_patterns),
            recording_targets=self.recommend_recording_targets(
                task_type, modality, species
            ),
            behavioral_measures=self.recommend_behavioral_measures(
                task_type
            ),
            analysis_pipeline=self.recommend_analysis_pipeline(
                task_type, modality
            ),
            reference_datasets=reference_datasets[:5],
            evidence=self.compile_evidence(design_params),
        )

        return recommendations

    def extract_design_parameters(self, dataset) -> DesignParameters:
        """
        Extract experimental design parameters from dataset metadata.

        Parameters extracted:
        - Trial timing (ITI, stimulus duration, response window)
        - Trial counts per condition
        - Session structure
        - Training progression
        - Behavioral criteria
        """
        params = DesignParameters()

        # From trial table
        if dataset.trials:
            params.trial_timing = self.analyze_trial_timing(dataset.trials)
            params.trial_counts = self.count_trials_by_type(dataset.trials)

        # From session metadata
        params.session_structure = self.analyze_sessions(dataset)

        # From task description
        params.task_phases = self.parse_task_phases(dataset.description)

        # From behavioral events
        params.behavioral_measures = self.identify_measures(dataset.events)

        return params

    def recommend_task_structure(
        self,
        task_type: str,
        common_patterns: Dict,
    ) -> TaskStructure:
        """
        Recommend task structure based on ontology and corpus evidence.
        """
        # Get base template from ontology
        base_template = self.ontology.get_task(task_type)

        structure = TaskStructure(
            phases=base_template.common_phases,
            events=base_template.common_events,
            conditions=self.infer_conditions(common_patterns),
        )

        # Annotate with corpus evidence
        structure.evidence = {
            "n_reference_datasets": len(common_patterns["datasets"]),
            "parameter_ranges": common_patterns["ranges"],
            "common_variants": common_patterns["variants"],
        }

        return structure
```

### Design Template Schema

```yaml
design_templates:
  reversal_learning:
    task_structure:
      phases:
        - name: "acquisition"
          description: "Initial learning of stimulus-reward contingencies"
          typical_trials: 50-100
        - name: "reversal"
          description: "Contingency switch"
          typical_trials: 50-100
          variants:
            - "deterministic reversal"
            - "probabilistic reversal"

    timing_parameters:
      inter_trial_interval:
        range: [2, 10]
        unit: "seconds"
        recommendation: "Variable ITI reduces timing strategies"
      stimulus_duration:
        range: [0.5, 2]
        unit: "seconds"
      response_window:
        range: [1, 5]
        unit: "seconds"

    behavioral_measures:
      - name: "choice"
        required: true
      - name: "reaction_time"
        required: true
      - name: "reward_outcome"
        required: true
      - name: "trial_outcome"
        required: true

    recommended_analyses:
      - q_learning_model_fitting
      - win_stay_lose_shift
      - perseveration_analysis
      - reversal_curve

  motor_imagery_bci:
    task_structure:
      phases:
        - name: "baseline"
          description: "Rest period before cue"
          duration: "2-3s"
        - name: "cue"
          description: "Instruction presentation"
          duration: "0.5-1s"
        - name: "imagery"
          description: "Motor imagery period"
          duration: "3-5s"
        - name: "feedback"
          description: "Optional feedback"
          duration: "1-2s"

    class_definitions:
      standard:
        - left_hand
        - right_hand
      extended:
        - left_hand
        - right_hand
        - feet
        - tongue

    recommended_analyses:
      - mu_beta_desynchronization
      - common_spatial_patterns
      - motor_imagery_classification
```

---

## 4. Benchmark/Audit Agent

### Purpose

Evaluate Neural Search system performance, audit corpus quality, and identify gaps or issues requiring attention.

### Capabilities

```yaml
benchmark_audit_agent:
  name: "BenchmarkAuditAgent"
  description: >
    Evaluates search quality, audits corpus coverage, and
    identifies quality issues.

  capabilities:
    # Search evaluation
    - run_benchmark_queries
    - compute_precision_recall
    - analyze_failure_cases
    - compare_to_baseline

    # Corpus audit
    - assess_coverage_by_domain
    - identify_metadata_gaps
    - detect_duplicate_records
    - validate_ontology_mapping

    # Quality assessment
    - score_extraction_confidence
    - flag_low_quality_records
    - prioritize_review_queue
    - track_quality_trends

    # Reporting
    - generate_benchmark_report
    - generate_coverage_report
    - generate_quality_report
    - visualize_metrics

  inputs:
    - benchmark_queries: str  # path to benchmark file
    - corpus_path: str
    - baseline_results: Optional[str]

  outputs:
    - benchmark_results: BenchmarkResults
    - coverage_report: CoverageReport
    - quality_report: QualityReport
    - recommended_actions: List[Action]
```

### Implementation Specification

```python
class BenchmarkAuditAgent:
    """
    Agent for evaluating and auditing Neural Search.
    """

    def __init__(self, search_engine, corpus, ontology):
        self.search = search_engine
        self.corpus = corpus
        self.ontology = ontology

    def run_benchmark(
        self,
        benchmark_path: str,
        baseline_path: str = None,
    ) -> BenchmarkResults:
        """
        Run benchmark evaluation suite.
        """
        benchmark = self.load_benchmark(benchmark_path)
        results = []

        for query in benchmark.queries:
            # Execute search
            search_results = self.search.query(
                query.query,
                constraints=query.required_constraints,
                num_results=10,
            )

            # Evaluate against expectations
            evaluation = self.evaluate_query(query, search_results)
            results.append(evaluation)

        # Aggregate metrics
        metrics = self.compute_aggregate_metrics(results)

        # Compare to baseline if provided
        if baseline_path:
            baseline = self.load_baseline(baseline_path)
            comparison = self.compare_to_baseline(metrics, baseline)
        else:
            comparison = None

        # Analyze failures
        failures = self.analyze_failures(results)

        return BenchmarkResults(
            individual_results=results,
            aggregate_metrics=metrics,
            baseline_comparison=comparison,
            failure_analysis=failures,
        )

    def evaluate_query(
        self,
        query: BenchmarkQuery,
        results: List[SearchResult],
    ) -> QueryEvaluation:
        """
        Evaluate search results against query expectations.
        """
        evaluation = QueryEvaluation(query_id=query.id)

        # Precision at K
        relevant_at_5 = sum(
            1 for r in results[:5]
            if self.is_relevant(r, query)
        )
        evaluation.precision_at_5 = relevant_at_5 / 5

        # Label recall
        found_tasks = set()
        found_modalities = set()
        found_regions = set()

        for result in results[:10]:
            found_tasks.update(result.tasks)
            found_modalities.update(result.modalities)
            found_regions.update(result.brain_regions)

        evaluation.task_recall = len(
            found_tasks & set(query.expected_tasks)
        ) / len(query.expected_tasks)

        # Constraint satisfaction (for adversarial queries)
        if query.required_constraints:
            evaluation.constraint_satisfaction = self.check_constraints(
                results, query.required_constraints
            )

        # Hard negative rejection
        if query.hard_negatives:
            evaluation.hard_negative_rejection = self.check_hard_negatives(
                results, query.hard_negatives
            )

        return evaluation

    def audit_corpus_coverage(self) -> CoverageReport:
        """
        Audit corpus coverage across ontology domains.
        """
        coverage = {}

        # Coverage by task
        for task in self.ontology.tasks:
            matching = self.corpus.count(task=task.id)
            coverage[f"task:{task.id}"] = {
                "count": matching,
                "target": task.target_count,
                "percent": matching / task.target_count * 100,
            }

        # Coverage by modality
        for modality in self.ontology.modalities:
            matching = self.corpus.count(modality=modality.id)
            coverage[f"modality:{modality.id}"] = {
                "count": matching,
                "has_coverage": matching > 0,
            }

        # Coverage by species
        for species in self.ontology.species:
            matching = self.corpus.count(species=species.id)
            coverage[f"species:{species.id}"] = {
                "count": matching,
            }

        # Identify gaps
        gaps = self.identify_coverage_gaps(coverage)

        return CoverageReport(
            coverage_matrix=coverage,
            gaps=gaps,
            recommendations=self.generate_coverage_recommendations(gaps),
        )

    def audit_data_quality(self) -> QualityReport:
        """
        Audit data quality across corpus.
        """
        quality_issues = []

        for record in self.corpus.all():
            issues = self.check_record_quality(record)
            if issues:
                quality_issues.append({
                    "record_id": record.id,
                    "issues": issues,
                    "severity": max(i.severity for i in issues),
                })

        return QualityReport(
            total_records=len(self.corpus),
            records_with_issues=len(quality_issues),
            issues_by_type=self.aggregate_issues(quality_issues),
            priority_queue=self.prioritize_for_review(quality_issues),
        )

    def check_record_quality(self, record) -> List[QualityIssue]:
        """
        Check individual record for quality issues.
        """
        issues = []

        # Missing required fields
        for field in ["title", "description", "species", "modality"]:
            if not getattr(record, field, None):
                issues.append(QualityIssue(
                    type="missing_required_field",
                    field=field,
                    severity="high",
                ))

        # Low extraction confidence
        if record.extraction_confidence < 0.6:
            issues.append(QualityIssue(
                type="low_confidence",
                value=record.extraction_confidence,
                severity="medium",
            ))

        # Ontology mapping issues
        unmapped_tasks = self.check_ontology_mapping(record.tasks)
        if unmapped_tasks:
            issues.append(QualityIssue(
                type="unmapped_ontology_term",
                terms=unmapped_tasks,
                severity="low",
            ))

        # Broken links
        if record.url and not self.check_url_accessible(record.url):
            issues.append(QualityIssue(
                type="broken_url",
                url=record.url,
                severity="high",
            ))

        return issues
```

### Benchmark Metrics Schema

```yaml
benchmark_metrics:
  precision_metrics:
    - name: precision_at_5
      formula: "relevant_in_top_5 / 5"
      target_easy: 0.8
      target_hard: 0.5

    - name: precision_at_10
      formula: "relevant_in_top_10 / 10"
      target_easy: 0.7
      target_hard: 0.4

  recall_metrics:
    - name: task_recall_at_10
      formula: "found_expected_tasks / expected_tasks"
      target: 0.8

    - name: dataset_recall
      formula: "found_expected_datasets / expected_datasets"
      target: 0.9  # when specific IDs expected

  ranking_metrics:
    - name: mrr
      formula: "1 / rank_of_first_relevant"
      target: 0.6

    - name: ndcg_at_10
      formula: "dcg_at_10 / idcg_at_10"
      target: 0.6

  constraint_metrics:
    - name: constraint_satisfaction
      formula: "results_satisfying_all_constraints / total_results"
      target: 1.0  # adversarial queries

    - name: hard_negative_rejection
      formula: "hard_negatives_not_returned / total_hard_negatives"
      target: 0.95
```

---

## 5. Notebook Generation Agent

### Purpose

Generate analysis notebooks tailored to specific datasets, including data loading, preprocessing, and analysis scaffolds appropriate for the dataset's modality and task.

### Capabilities

```yaml
notebook_generation_agent:
  name: "NotebookGenerationAgent"
  description: >
    Generates starter analysis notebooks customized to specific
    datasets and analysis goals.

  capabilities:
    # Template selection
    - select_modality_template
    - select_task_template
    - customize_for_dataset

    # Code generation
    - generate_data_loading_code
    - generate_preprocessing_code
    - generate_analysis_code
    - generate_visualization_code

    # Documentation
    - generate_dataset_summary
    - generate_analysis_rationale
    - generate_interpretation_guide

    # Validation
    - validate_generated_code
    - check_dependency_availability
    - estimate_computational_requirements

  inputs:
    - dataset_id: str
    - analysis_type: str  # e.g., "q_learning_modeling", "choice_decoding"
    - output_format: Literal["jupyter", "colab", "python"]
    - include_plots: bool = True

  outputs:
    - notebook: NotebookContent
    - dependencies: List[str]
    - estimated_runtime: str
    - data_size_warning: Optional[str]
```

### Implementation Specification

```python
class NotebookGenerationAgent:
    """
    Agent for generating analysis notebooks.
    """

    def __init__(self, corpus, templates, ontology):
        self.corpus = corpus
        self.templates = templates
        self.ontology = ontology

    def generate_notebook(
        self,
        dataset_id: str,
        analysis_type: str,
        output_format: str = "jupyter",
    ) -> GeneratedNotebook:
        """
        Generate analysis notebook for dataset.

        Steps:
        1. Load dataset metadata
        2. Select appropriate template
        3. Customize template for dataset
        4. Generate data loading code
        5. Generate analysis code
        6. Add documentation
        7. Validate and return
        """

        # Step 1: Load dataset
        dataset = self.corpus.get(dataset_id)

        # Step 2: Select template
        template = self.select_template(
            modality=dataset.modality,
            task=dataset.task,
            analysis=analysis_type,
        )

        # Step 3: Customize
        customized = self.customize_template(template, dataset, analysis_type)

        # Step 4: Data loading
        data_loading = self.generate_data_loading(dataset)

        # Step 5: Analysis code
        analysis_code = self.generate_analysis_code(
            dataset, analysis_type, template
        )

        # Step 6: Documentation
        docs = self.generate_documentation(dataset, analysis_type)

        # Step 7: Assemble notebook
        notebook = self.assemble_notebook(
            metadata=docs.metadata_section,
            data_loading=data_loading,
            preprocessing=customized.preprocessing,
            analysis=analysis_code,
            visualization=customized.visualization,
            interpretation=docs.interpretation_guide,
            format=output_format,
        )

        # Validate
        validation = self.validate_notebook(notebook)

        return GeneratedNotebook(
            content=notebook,
            dependencies=self.extract_dependencies(notebook),
            validation=validation,
            warnings=self.generate_warnings(dataset, analysis_type),
        )

    def select_template(
        self,
        modality: str,
        task: str,
        analysis: str,
    ) -> NotebookTemplate:
        """
        Select best matching template.

        Priority:
        1. Exact match (modality + task + analysis)
        2. Modality + analysis match
        3. Analysis type match
        4. Modality match (generic)
        """
        # Try exact match
        exact_key = f"{modality}_{task}_{analysis}"
        if exact_key in self.templates:
            return self.templates[exact_key]

        # Try modality + analysis
        mod_analysis_key = f"{modality}_{analysis}"
        if mod_analysis_key in self.templates:
            return self.templates[mod_analysis_key]

        # Try analysis only
        if analysis in self.templates:
            return self.templates[analysis]

        # Fall back to modality generic
        return self.templates.get(modality, self.templates["generic"])

    def generate_data_loading(self, dataset) -> str:
        """
        Generate data loading code based on data standard.
        """
        if dataset.data_standard == "NWB":
            return self.generate_nwb_loading(dataset)
        elif dataset.data_standard == "BIDS":
            return self.generate_bids_loading(dataset)
        else:
            return self.generate_generic_loading(dataset)

    def generate_nwb_loading(self, dataset) -> str:
        """
        Generate NWB-specific loading code.
        """
        code = f'''
# Load NWB file
from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd

# Dataset: {dataset.name}
# Source: {dataset.url}

nwb_path = "path/to/your/file.nwb"  # TODO: Update with actual path

with NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()

    # Session info
    print(f"Session: {{nwb.identifier}}")
    print(f"Description: {{nwb.session_description}}")

    # Extract neural data
    # TODO: Customize based on acquisition type
'''
        # Add modality-specific loading
        if "ephys" in dataset.modality or "neuropixels" in dataset.modality:
            code += self.generate_ephys_loading()
        elif "calcium" in dataset.modality:
            code += self.generate_calcium_loading()

        # Add trial loading if available
        if dataset.has_trials:
            code += self.generate_trial_loading()

        return code

    def generate_analysis_code(
        self,
        dataset,
        analysis_type: str,
        template,
    ) -> str:
        """
        Generate analysis-specific code.
        """
        if analysis_type == "q_learning_model_fitting":
            return self.generate_qlearning_code(dataset)
        elif analysis_type == "choice_decoding":
            return self.generate_decoding_code(dataset, "choice")
        elif analysis_type == "motor_decoding":
            return self.generate_decoding_code(dataset, "motor")
        elif analysis_type == "event_aligned_activity":
            return self.generate_event_aligned_code(dataset)
        elif analysis_type == "population_dynamics":
            return self.generate_population_dynamics_code(dataset)
        else:
            return template.default_analysis_code
```

### Notebook Template Schema

```yaml
notebook_templates:
  neuropixels_choice_decoding:
    title: "Choice Decoding from Neuropixels Data"
    description: >
      Decode binary choices from neural population activity
      using cross-validated logistic regression.

    sections:
      - name: "Setup"
        type: "code"
        content: |
          import numpy as np
          import pandas as pd
          from sklearn.linear_model import LogisticRegressionCV
          from sklearn.model_selection import StratifiedKFold
          import matplotlib.pyplot as plt

      - name: "Data Loading"
        type: "template"
        template_key: "nwb_ephys_loading"

      - name: "Preprocessing"
        type: "code"
        content: |
          # Bin spikes into time windows
          def bin_spikes(spike_times, bin_edges):
              # TODO: Customize bin size
              pass

          # Align to trial events
          def align_to_trials(neural_data, trial_times, pre_time, post_time):
              # TODO: Customize alignment
              pass

      - name: "Decoding Analysis"
        type: "code"
        content: |
          # Choice decoding with cross-validation
          def decode_choice(X, y, n_folds=5):
              cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
              model = LogisticRegressionCV(cv=cv)
              model.fit(X, y)
              return model.score(X, y), model

          # Run decoding
          accuracy, model = decode_choice(neural_data, choice_labels)
          print(f"Decoding accuracy: {accuracy:.2%}")

      - name: "Visualization"
        type: "code"
        content: |
          # Plot results
          fig, ax = plt.subplots(1, 2, figsize=(12, 5))
          # TODO: Add visualization code

    dependencies:
      - pynwb
      - numpy
      - pandas
      - scikit-learn
      - matplotlib

  fiber_photometry_td_error:
    title: "Temporal Difference Error Analysis from Fiber Photometry"
    # ... similar structure
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

```yaml
phase_1:
  - task: "Core agent interfaces"
    deliverables:
      - Agent base class
      - Input/output schemas
      - Error handling

  - task: "Dataset Discovery Agent MVP"
    deliverables:
      - Query parsing
      - Hybrid retrieval
      - Basic ranking

  - task: "Notebook Generation Agent MVP"
    deliverables:
      - NWB loading templates
      - 3 analysis templates
      - Basic customization
```

### Phase 2: Integration (Weeks 3-4)

```yaml
phase_2:
  - task: "Paper-Dataset Linking Agent"
    deliverables:
      - OpenAlex integration
      - Link evidence scoring
      - Bidirectional linking

  - task: "Benchmark Agent"
    deliverables:
      - Benchmark runner
      - Metrics computation
      - Coverage auditing

  - task: "Experimental Design Agent MVP"
    deliverables:
      - Parameter extraction
      - 5 task design templates
      - Recommendations
```

### Phase 3: Refinement (Weeks 5-6)

```yaml
phase_3:
  - task: "Agent orchestration"
    deliverables:
      - Multi-agent workflows
      - Context passing
      - Result aggregation

  - task: "LLM integration"
    deliverables:
      - Query parsing with LLM
      - Natural language explanations
      - Conversation support

  - task: "Quality improvements"
    deliverables:
      - Failure case handling
      - Confidence calibration
      - User feedback loop
```

---

## Agent Communication Protocol

### Message Schema

```python
@dataclass
class AgentMessage:
    """Standard message format for agent communication."""

    sender: str  # Agent ID
    recipient: str  # Agent ID or "user"
    message_type: Literal[
        "query",
        "result",
        "request",
        "response",
        "error",
        "status",
    ]
    content: Dict[str, Any]
    context: Dict[str, Any]  # Shared context
    timestamp: datetime
    correlation_id: str  # For tracking conversations
```

### Workflow Example

```yaml
example_workflow:
  name: "Research Question to Notebook"
  description: >
    User asks a research question, agents collaborate to
    find datasets and generate analysis notebook.

  steps:
    1:
      agent: DatasetDiscoveryAgent
      input: "dopamine recordings during reversal learning"
      output: ranked_datasets

    2:
      agent: PaperDatasetLinkingAgent
      input: top_dataset_from_step_1
      output: linked_papers

    3:
      agent: ExperimentalDesignAgent
      input: {dataset: top_dataset, task: reversal_learning}
      output: design_reference

    4:
      agent: NotebookGenerationAgent
      input:
        dataset: top_dataset
        analysis: q_learning_model_fitting
        context: {papers: linked_papers, design: design_reference}
      output: generated_notebook

    5:
      agent: BenchmarkAuditAgent
      input: {query: original_query, results: datasets}
      output: quality_assessment
      purpose: "Log for system improvement"
```
