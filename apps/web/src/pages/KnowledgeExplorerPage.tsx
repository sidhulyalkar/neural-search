import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchGalaxyLayout, fetchSubgraph, fetchTopicGraph } from '../api/graph'
import { GalaxyGraph } from '../components/graph/GalaxyGraph'
import { FilterPanel } from '../components/graph/FilterPanel'
import { SuggestedViews } from '../components/graph/SuggestedViews'
import { OntologyTreePanel } from '../components/graph/OntologyTreePanel'
import { NodeDetailPanel } from '../components/graph/NodeDetailPanel'
import { GraphLegend } from '../components/graph/GraphLegend'
import { TopicsPanel } from '../components/graph/TopicsPanel'
import type { GalaxyPoint, GraphNode, LayerMode, NodeType } from '../types/graph'

// Which node type each layer spotlights (dims all other types)
const LAYER_TYPE: Partial<Record<LayerMode, NodeType>> = {
  corpus: 'dataset',
  literature: 'paper',
  consensus: 'finding_cluster',
  morphology: 'region',
}

const LAYER_MODES: { value: LayerMode; label: string; title: string }[] = [
  { value: 'corpus',        label: 'Corpus',    title: 'Dataset Corpus — 7,171 dataset records' },
  { value: 'consensus',     label: 'Consensus', title: 'Consensus Findings — aggregated evidence per region' },
  { value: 'literature',    label: 'Literature',title: 'Literature Findings — ~12K extracted findings' },
  { value: 'bridge',        label: 'Bridge',    title: 'Paper-Dataset Bridges' },
  { value: 'morphology',    label: 'Morphology',title: 'Structural connectivity and anatomy' },
  { value: 'validation',    label: 'Validation',title: 'Validation / Qrels' },
  { value: 'coverage_gaps', label: 'Gaps',      title: 'Coverage Gaps — under-studied regions' },
  { value: 'topics',        label: 'Topics',    title: 'Topic Explorer — 26 canonical topics' },
]

export function KnowledgeExplorerPage() {
  const [layerMode, setLayerMode] = useState<LayerMode>('corpus')
  const [regionFilter, setRegionFilter] = useState<string[]>([])
  const [activeTopicSlug, setActiveTopicSlug] = useState<string | null>(null)
  const [selectedPoint, setSelectedPoint] = useState<GalaxyPoint | null>(null)
  const [legendOpen, setLegendOpen] = useState(false)
  const [companionSlugs, setCompanionSlugs] = useState<string[]>([])

  const [searchParams] = useSearchParams()

  useEffect(() => {
    const region = searchParams.get('region')
    if (region) setRegionFilter([region])
  }, [searchParams])

  const { data: galaxyLayout } = useQuery({
    queryKey: ['galaxy-layout'],
    queryFn: fetchGalaxyLayout,
    staleTime: Infinity,
  })

  const { data: subgraphData } = useQuery({
    queryKey: ['graph-subgraph', regionFilter],
    queryFn: () => fetchSubgraph({ regions: regionFilter.join(',') }),
    enabled: regionFilter.length > 0 && !activeTopicSlug,
    staleTime: 30_000,
  })

  const { data: topicData } = useQuery({
    queryKey: ['graph-topic', activeTopicSlug],
    queryFn: () => fetchTopicGraph(activeTopicSlug!),
    enabled: Boolean(activeTopicSlug),
    staleTime: 60_000,
  })

  useEffect(() => {
    setCompanionSlugs(topicData?.topic?.companion_slugs ?? [])
  }, [topicData])

  const highlightIds = useMemo<Set<string> | undefined>(() => {
    const points = galaxyLayout?.nodes ?? []

    if (activeTopicSlug && topicData) {
      const ids = new Set(topicData.nodes.map((n) => n.id))
      return ids.size > 0 ? ids : undefined
    }

    if (regionFilter.length > 0 && subgraphData) {
      const ids = new Set(subgraphData.nodes.map((n) => n.id))
      return ids.size > 0 ? ids : undefined
    }

    const layerType = LAYER_TYPE[layerMode]
    if (layerType) {
      return new Set(points.filter((p) => p.type === layerType).map((p) => p.id))
    }

    return undefined
  }, [activeTopicSlug, topicData, regionFilter, subgraphData, layerMode, galaxyLayout])

  const handleTopicSelect = useCallback((slug: string) => {
    setActiveTopicSlug(slug)
    setRegionFilter([])
  }, [])

  const handleFiltersChange = useCallback(
    (f: { regions: string[]; species: string[]; tasks: string[] }) => {
      setRegionFilter(f.regions)
      setActiveTopicSlug(null)
    },
    [],
  )

  const handleRegionSelect = useCallback((region: string) => {
    setRegionFilter((prev) => (prev.includes(region) ? prev : [...prev, region]))
    setActiveTopicSlug(null)
  }, [])

  const handlePointClick = useCallback(
    (id: string, label: string) => {
      const p = galaxyLayout?.nodes.find((n) => n.id === id) ?? null
      setSelectedPoint(
        p ?? { id, label, x: 0, y: 0, z: 0, type: 'region', color: '#94a3b8', size: 5 },
      )
    },
    [galaxyLayout],
  )

  const selectedNode = useMemo<GraphNode | null>(() => {
    if (!selectedPoint) return null
    return {
      id: selectedPoint.id,
      type: selectedPoint.type,
      label: selectedPoint.label,
      scale_level: 3,
      size: selectedPoint.size,
      color: selectedPoint.color,
      meta: {},
    }
  }, [selectedPoint])

  const hasFilters = regionFilter.length > 0 || Boolean(activeTopicSlug)
  const highlightCount = highlightIds?.size ?? 0

  return (
    <div className="relative w-full" style={{ height: 'calc(100vh - 56px)' }}>

      {/* Layer mode bar */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-0.5 bg-neural-950/80 backdrop-blur border border-neural-800/50 rounded-lg px-3 py-2">
        {LAYER_MODES.map(({ value, label, title }) => (
          <button
            key={value}
            type="button"
            onClick={() => {
              setLayerMode(value)
              setActiveTopicSlug(null)
              setRegionFilter([])
            }}
            title={title}
            className={`px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${
              layerMode === value
                ? 'bg-accent-violet/20 text-accent-violet border border-accent-violet/30'
                : 'text-neural-500 hover:text-neural-200 border border-transparent'
            }`}
          >
            {label}
          </button>
        ))}
        <button
          type="button"
          onClick={() => setLegendOpen((v) => !v)}
          className="ml-2 text-xs text-neural-500 hover:text-neural-200 px-2 py-1.5 rounded border border-transparent hover:border-neural-700 transition-colors"
          title="Toggle legend"
        >
          ?
        </button>
      </div>

      {legendOpen && <GraphLegend onClose={() => setLegendOpen(false)} />}

      {/* Left sidebar */}
      <div className="absolute top-0 left-0 bottom-0 z-10 w-52 bg-neural-950/80 backdrop-blur border-r border-neural-800/40 p-4 overflow-y-auto flex flex-col gap-6">
        <FilterPanel
          regions={regionFilter}
          species={[]}
          tasks={[]}
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

      {/* Galaxy canvas */}
      <div className="absolute top-0 left-52 right-0 bottom-0">
        <GalaxyGraph
          points={galaxyLayout?.nodes ?? []}
          highlightIds={highlightIds}
          onPointClick={handlePointClick}
        />
      </div>

      {/* Status bar */}
      <div className="absolute top-14 right-4 z-10 flex items-center gap-2">
        <span className="text-xs text-neural-500 bg-neural-950/80 backdrop-blur border border-neural-800/40 rounded px-3 py-1.5">
          {hasFilters
            ? `${highlightCount} matching · ${galaxyLayout?.nodes.length ?? 0} total`
            : `${galaxyLayout?.nodes.length ?? 0} nodes · ${layerMode}`}
          {activeTopicSlug && topicData?.topic && ` · ${topicData.topic.label}`}
        </span>
        {hasFilters && (
          <button
            type="button"
            onClick={() => {
              setRegionFilter([])
              setActiveTopicSlug(null)
              setCompanionSlugs([])
              setSelectedPoint(null)
            }}
            className="text-xs text-neural-600 hover:text-neural-300 bg-neural-950/80 backdrop-blur border border-neural-800/40 rounded px-3 py-1.5 transition-colors"
          >
            Reset
          </button>
        )}
      </div>

      <NodeDetailPanel
        node={selectedNode}
        findings={[]}
        consensus={[]}
        onClose={() => setSelectedPoint(null)}
      />

      {!galaxyLayout && (
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
