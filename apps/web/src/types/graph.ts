export type NodeType = 'system' | 'region' | 'finding_cluster' | 'dataset' | 'paper'
export type EdgeType = 'contains' | 'covers' | 'linked' | 'supports' | 'contradicts'
export type ViewMode = 'galaxy' | 'explorer' | '2d'
export type LayerMode = 'corpus' | 'consensus' | 'literature' | 'bridge' | 'morphology' | 'validation' | 'coverage_gaps' | 'topics'
export type Direction = 'increase' | 'decrease' | 'correlation' | 'no_change'

export interface GraphNode {
  id: string
  type: NodeType
  label: string
  scale_level: number    // 0=molecule, 2=cluster, 3=region, 4=system/dataset
  size: number
  color: string
  meta: Record<string, unknown>
}

export interface GraphEdge {
  source: string
  target: string
  type: EdgeType
  weight: number
  color: string
}

// react-force-graph uses 'links', not 'edges'
export interface GraphData {
  nodes: GraphNode[]
  links: GraphEdge[]
}

export interface GalaxyPoint {
  id: string
  x: number
  y: number
  z: number
  type: NodeType
  color: string
  size: number
  label: string
}

export interface GalaxyLayout {
  nodes: GalaxyPoint[]
}

export interface SubgraphResponse {
  nodes: GraphNode[]
  links: GraphEdge[]
  meta: { node_count: number; edge_count: number; filtered_by: Record<string, unknown> }
}

export interface TopicGraphResponse extends SubgraphResponse {
  topic: { slug: string; label: string; description: string; companion_slugs: string[] }
}

export interface SuggestedView {
  slug: string
  label: string
  description: string
  layer: LayerMode
  companions: string[]
}

export interface ConsensusRow {
  region: string
  direction: Direction
  task: string | null
  n_findings: number
  n_papers: number
  consensus_strength: number
}

export interface FindingRow {
  finding_id: string
  finding_text: string
  region: string
  direction: Direction
  confidence: number
  paper_id: string
}

export interface DatasetNeighborhood {
  dataset_id: string
  linked_papers: Array<{
    paper_openalex_id: string
    paper_title: string | null
    paper_year: number | null
    paper_doi: string | null
    confidence: number
  }>
  finding_clusters: ConsensusRow[]
  related_datasets: Array<{
    dataset_id: string
    title: string
    shared_regions: string[]
  }>
  consensus_by_region: ConsensusRow[]
}

// Timeline types (added by Phase 4)
export interface TimelineYear {
  year: number
  n_papers: number
  n_findings: number
  n_datasets: number
  methods_introduced: string[]
  top_papers: Array<{ id: string; title: string; citation_count: number }>
  top_findings: Array<{ text: string; region: string; direction: string }>
}

export interface TopicTimeline {
  topic_id: string
  topic_label: string
  entries: TimelineYear[]
  total_papers: number
  total_findings: number
  total_datasets: number
  year_range: [number, number] | null
  key_regions: string[]
  key_methods: string[]
}

export interface TopicSummary {
  id: string
  label: string
  description: string
  color: string
  companion_topics: string[]
  n_tasks: number
  n_regions: number
}

export interface FoundationalPaper {
  id: string
  title: string
  year: number | null
  doi: string | null
  citation_count: number
  in_topic_citations: number
}
