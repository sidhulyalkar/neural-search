import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  fetchFindings,
  fetchGalaxyLayout,
  fetchGraphOverview,
  fetchSubgraph,
  fetchTopicGraph,
} from '../api/graph'
import { GalaxyGraph } from '../components/graph/GalaxyGraph'
import { ExplorerGraph } from '../components/graph/ExplorerGraph'
import { GraphControls } from '../components/graph/GraphControls'
import { FilterPanel } from '../components/graph/FilterPanel'
import { SuggestedViews } from '../components/graph/SuggestedViews'
import { OntologyTreePanel } from '../components/graph/OntologyTreePanel'
import { NodeDetailPanel } from '../components/graph/NodeDetailPanel'
import { GraphLegend } from '../components/graph/GraphLegend'
import { TopicsPanel } from '../components/graph/TopicsPanel'
import type { GraphData, GraphNode, LayerMode, ViewMode } from '../types/graph'

interface ActiveFilters {
  regions: string[]
  species: string[]
  tasks: string[]
}

const EMPTY_FILTERS: ActiveFilters = { regions: [], species: [], tasks: [] }

function filtersToParams(f: ActiveFilters) {
  return {
    regions: f.regions.join(','),
    species: f.species.join(','),
    tasks: f.tasks.join(','),
  }
}

function hasFilters(f: ActiveFilters) {
  return f.regions.length > 0 || f.species.length > 0 || f.tasks.length > 0
}

export function KnowledgeExplorerPage() {
  const [viewMode, setViewMode] = useState<ViewMode>('galaxy')
  const [layerMode, setLayerMode] = useState<LayerMode>('corpus')
  const [filters, setFilters] = useState<ActiveFilters>(EMPTY_FILTERS)
  const [activeTopicSlug, setActiveTopicSlug] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [legendOpen, setLegendOpen] = useState(false)
  const [companionSlugs, setCompanionSlugs] = useState<string[]>([])

  const [searchParams] = useSearchParams()

  useEffect(() => {
    const datasetParam = searchParams.get('dataset')
    if (!datasetParam) return
    setViewMode('explorer')
    setSelectedNode({
      id: `dataset:${datasetParam}`,
      type: 'dataset',
      label: datasetParam,
      scale_level: 3,
      size: 10,
      color: '#22d3ee',
      meta: { source: '', readiness: 0, brain_regions: [] },
    })
  }, [searchParams])

  // Galaxy layout (static file, loads once)
  const { data: galaxyLayout } = useQuery({
    queryKey: ['galaxy-layout'],
    queryFn: fetchGalaxyLayout,
    staleTime: Infinity,
  })

  // Overview graph (no filters)
  const { data: overviewData } = useQuery({
    queryKey: ['graph-overview'],
    queryFn: () => fetchGraphOverview(400),
    staleTime: 60_000,
  })

  // Subgraph (when filters active)
  const { data: subgraphData } = useQuery({
    queryKey: ['graph-subgraph', filters],
    queryFn: () => fetchSubgraph(filtersToParams(filters)),
    enabled: hasFilters(filters) && !activeTopicSlug,
    staleTime: 30_000,
  })

  // Topic graph
  const { data: topicData } = useQuery({
    queryKey: ['graph-topic', activeTopicSlug],
    queryFn: () => fetchTopicGraph(activeTopicSlug!),
    enabled: Boolean(activeTopicSlug),
    staleTime: 60_000,
  })

  // Findings for selected node
  const { data: nodefindings = [] } = useQuery({
    queryKey: ['node-findings', selectedNode?.id],
    queryFn: () => {
      const region = (selectedNode?.meta?.region as string) ?? selectedNode?.label ?? ''
      return fetchFindings({ region, limit: 5 })
    },
    enabled: Boolean(
      selectedNode &&
        (selectedNode.type === 'finding_cluster' || selectedNode.type === 'region'),
    ),
    staleTime: 60_000,
  })

  const graphData = useMemo<GraphData>(() => {
    const raw = activeTopicSlug
      ? topicData
      : hasFilters(filters)
      ? subgraphData
      : overviewData
    if (!raw) return { nodes: [], links: [] }
    return { nodes: raw.nodes, links: raw.links }
  }, [activeTopicSlug, topicData, filters, subgraphData, overviewData])

  // When topic data arrives, update companion slugs
  useEffect(() => {
    if (topicData?.topic?.companion_slugs) {
      setCompanionSlugs(topicData.topic.companion_slugs)
    } else {
      setCompanionSlugs([])
    }
  }, [topicData])

  const handleTopicSelect = useCallback((slug: string) => {
    setActiveTopicSlug(slug)
    setFilters(EMPTY_FILTERS)
    setViewMode('explorer')
  }, [])

  const handleFiltersChange = useCallback((f: ActiveFilters) => {
    setFilters(f)
    setActiveTopicSlug(null)
    if (hasFilters(f)) setViewMode('explorer')
  }, [])

  const handleRegionSelect = useCallback((region: string) => {
    setFilters((prev) => ({
      ...prev,
      regions: prev.regions.includes(region) ? prev.regions : [...prev.regions, region],
    }))
    setActiveTopicSlug(null)
    setViewMode('explorer')
  }, [])

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node)
  }, [])

  const handleGalaxyPointClick = useCallback((id: string, _label: string) => {
    // Extract region from id (e.g. "region:hippocampus" → "hippocampus")
    const parts = id.split(':')
    if (parts[0] === 'region' || parts[0] === 'cluster') {
      const region = parts[1]
      setFilters({ regions: [region], species: [], tasks: [] })
      setActiveTopicSlug(null)
      setViewMode('explorer')
    }
  }, [])

  return (
    <div className="relative w-full" style={{ height: 'calc(100vh - 56px)' }}>
      {/* Top controls */}
      <GraphControls
        viewMode={viewMode}
        layerMode={layerMode}
        onViewModeChange={setViewMode}
        onLayerModeChange={setLayerMode}
        onLegendToggle={() => setLegendOpen((v) => !v)}
      />

      {/* Legend overlay */}
      {legendOpen && <GraphLegend onClose={() => setLegendOpen(false)} />}

      {/* Left sidebar */}
      <div className="absolute top-0 left-0 bottom-0 z-10 w-52 bg-neural-950/80 backdrop-blur border-r border-neural-800/40 p-4 overflow-y-auto flex flex-col gap-6">
        <FilterPanel
          regions={filters.regions}
          species={filters.species}
          tasks={filters.tasks}
          onFiltersChange={handleFiltersChange}
        />

        {layerMode === 'topics' ? (
          <TopicsPanel activeSlug={activeTopicSlug} onTopicSelect={handleTopicSelect} />
        ) : (
          <SuggestedViews
            activeSlug={activeTopicSlug}
            companionSlugs={companionSlugs}
            onViewSelect={handleTopicSelect}
          />
        )}

        <OntologyTreePanel onRegionSelect={handleRegionSelect} />
      </div>

      {/* Canvas area */}
      <div className="absolute top-0 left-52 right-0 bottom-0">
        {viewMode === 'galaxy' && (
          <GalaxyGraph
            points={galaxyLayout?.nodes ?? []}
            onPointClick={handleGalaxyPointClick}
          />
        )}
        {(viewMode === 'explorer' || viewMode === '2d') && (
          <ExplorerGraph
            graphData={graphData}
            mode={viewMode === '2d' ? '2d' : '3d'}
            onNodeClick={handleNodeClick}
          />
        )}
      </div>

      {/* Status bar */}
      {(activeTopicSlug || hasFilters(filters)) && (
        <div className="absolute top-14 right-4 z-10 flex items-center gap-2">
          <span className="text-xs text-neural-500 bg-neural-950/80 backdrop-blur border border-neural-800/40 rounded px-3 py-1.5">
            {graphData.nodes.length} nodes · {graphData.links.length} links
            {activeTopicSlug && topicData?.topic && ` · ${topicData.topic.label}`}
          </span>
          <button
            type="button"
            onClick={() => {
              setFilters(EMPTY_FILTERS)
              setActiveTopicSlug(null)
              setCompanionSlugs([])
              setSelectedNode(null)
              setViewMode('galaxy')
            }}
            className="text-xs text-neural-600 hover:text-neural-300 bg-neural-950/80 backdrop-blur border border-neural-800/40 rounded px-3 py-1.5 transition-colors"
          >
            Reset
          </button>
        </div>
      )}

      {/* Node detail panel */}
      <NodeDetailPanel
        node={selectedNode}
        findings={nodefindings}
        consensus={[]}
        onClose={() => setSelectedNode(null)}
      />

      {/* Loading states */}
      {viewMode === 'galaxy' && !galaxyLayout && (
        <div className="absolute inset-0 flex items-center justify-center bg-neural-950/50 z-30">
          <div className="flex items-center gap-3 text-neural-400 text-sm">
            <span className="w-4 h-4 border-2 border-neural-700 border-t-accent-cyan rounded-full animate-spin" />
            Loading galaxy…
          </div>
        </div>
      )}
    </div>
  )
}
