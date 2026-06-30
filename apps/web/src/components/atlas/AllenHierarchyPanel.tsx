import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  atlasApi,
  type AllenStructure,
} from '../../api/coverage'

// ── Types ────────────────────────────────────────────────────────────────────

export type AllenHierarchyPanelProps = {
  onRegionSelect: (regionId: string) => void
  countMap: Map<string, { n: number; label: string }>
  filterLobeRegionIds?: string[] | null
}

// ── Coverage dot ─────────────────────────────────────────────────────────────

function CoverageDot({ n }: { n: number }) {
  let color = '#334155'
  if (n >= 400) color = '#059669'
  else if (n >= 100) color = '#047857'
  else if (n >= 30) color = '#a16207'
  else if (n >= 5) color = '#c2410c'
  else if (n >= 1) color = '#991b1b'

  return (
    <span
      title={`${n} datasets`}
      className="inline-block w-2 h-2 rounded-full flex-shrink-0 ml-1"
      style={{ backgroundColor: color }}
    />
  )
}

// ── Tree node ────────────────────────────────────────────────────────────────

type TreeNodeProps = {
  structure: AllenStructure
  depth: number
  ontologyMapping: Record<string, number>
  countMap: Map<string, { n: number; label: string }>
  onRegionSelect: (regionId: string) => void
}

function TreeNode({ structure, depth, ontologyMapping, countMap, onRegionSelect }: TreeNodeProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const { data: children, isFetching } = useQuery({
    queryKey: ['allen-children', structure.allen_id],
    queryFn: () => atlasApi.children(structure.allen_id),
    enabled: isExpanded && structure.children_ids.length > 0,
    staleTime: Infinity,
  })

  // Find if this allen_id maps to any ontology region
  const ontologyId = Object.entries(ontologyMapping).find(
    ([, allenId]) => allenId === structure.allen_id
  )?.[0]

  const coverageEntry = ontologyId ? countMap.get(ontologyId) : undefined
  const datasetCount = coverageEntry?.n ?? 0

  const hasChildren = structure.children_ids.length > 0
  const indentPx = depth * 12

  function handleToggle() {
    if (hasChildren) setIsExpanded((prev) => !prev)
  }

  function handleSelect() {
    if (ontologyId) {
      onRegionSelect(ontologyId)
    }
  }

  return (
    <div>
      <div
        className={`flex items-center gap-1.5 py-0.5 pr-2 rounded text-xs group hover:bg-neural-800/50 cursor-pointer transition-colors ${
          ontologyId ? 'hover:text-neural-100' : ''
        }`}
        style={{ paddingLeft: `${indentPx + 4}px` }}
        onClick={ontologyId ? handleSelect : handleToggle}
        title={structure.name}
        role="treeitem"
        aria-expanded={hasChildren ? isExpanded : undefined}
      >
        {/* Expand/collapse chevron */}
        <span
          className={`text-neural-600 group-hover:text-neural-400 transition-colors flex-shrink-0 w-3 text-center ${
            hasChildren ? 'cursor-pointer' : 'opacity-0'
          }`}
          onClick={(e) => {
            e.stopPropagation()
            handleToggle()
          }}
        >
          {hasChildren ? (isExpanded ? '▾' : '▸') : '·'}
        </span>

        {/* Allen color swatch */}
        <span
          className="inline-block w-2 h-2 rounded-sm flex-shrink-0 border border-black/20"
          style={{ backgroundColor: `#${structure.color_hex}` }}
        />

        {/* Acronym */}
        <span className="font-mono text-neural-300 font-medium flex-shrink-0">
          {structure.acronym}
        </span>

        {/* Name — truncate */}
        <span className="text-neural-500 truncate min-w-0 group-hover:text-neural-400">
          {structure.name}
        </span>

        {/* Coverage dot */}
        {datasetCount > 0 && <CoverageDot n={datasetCount} />}

        {isFetching && (
          <span className="text-neural-700 text-xs ml-auto">…</span>
        )}
      </div>

      {/* Children */}
      {isExpanded && children && children.length > 0 && (
        <div role="group">
          {children.map((child) => (
            <TreeNode
              key={child.allen_id}
              structure={child}
              depth={depth + 1}
              ontologyMapping={ontologyMapping}
              countMap={countMap}
              onRegionSelect={onRegionSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

export function AllenHierarchyPanel({
  onRegionSelect,
  countMap,
  filterLobeRegionIds,
}: AllenHierarchyPanelProps) {
  const [searchQuery, setSearchQuery] = useState('')

  const { data: structuresData, isLoading: structuresLoading } = useQuery({
    queryKey: ['allen-structures', 'mouse', 5],
    queryFn: () => atlasApi.structures('mouse', 5),
    staleTime: 1000 * 60 * 10,
  })

  const { data: mappingData, isLoading: mappingLoading } = useQuery({
    queryKey: ['allen-mapping'],
    queryFn: atlasApi.mapping,
    staleTime: 1000 * 60 * 10,
  })

  const isLoading = structuresLoading || mappingLoading
  const ontologyMapping = mappingData?.mapping ?? {}

  // Filter structures by search or lobe filter
  const structures = structuresData?.structures ?? []
  const filtered = structures.filter((s) => {
    if (filterLobeRegionIds && filterLobeRegionIds.length > 0) {
      // Check if this structure's allen_id appears in the mapping for any of the lobe region ids
      const relevantAllenIds = new Set(
        filterLobeRegionIds
          .map((rid) => ontologyMapping[rid])
          .filter(Boolean)
      )
      if (relevantAllenIds.size > 0 && !relevantAllenIds.has(s.allen_id)) return false
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      return (
        s.acronym.toLowerCase().includes(q) ||
        s.name.toLowerCase().includes(q)
      )
    }
    return true
  })

  const mappedCount = Object.keys(ontologyMapping).length

  return (
    <div className="flex flex-col gap-2 h-full">
      {/* Header */}
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-mono text-neural-400 uppercase tracking-wider">
          Allen CCF Hierarchy
        </span>
        {!mappingLoading && (
          <span className="text-xs text-neural-600 font-mono">
            {mappedCount} mapped
          </span>
        )}
      </div>

      {/* Search */}
      <input
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder="Search regions…"
        className="w-full bg-neural-900 border border-neural-700 rounded px-2 py-1.5 text-xs text-neural-200 placeholder-neural-600 focus:outline-none focus:border-neural-500"
      />

      {/* Lobe filter badge */}
      {filterLobeRegionIds && filterLobeRegionIds.length > 0 && (
        <div className="text-xs text-accent-cyan border border-accent-cyan/30 bg-accent-cyan/5 rounded px-2 py-1">
          Filtered by lobe selection
        </div>
      )}

      {/* Tree */}
      <div className="overflow-y-auto flex-1 min-h-0" role="tree" aria-label="Allen Brain Atlas hierarchy">
        {isLoading && (
          <div className="text-neural-500 text-xs py-4 text-center">Loading hierarchy…</div>
        )}
        {!isLoading && filtered.length === 0 && (
          <div className="text-neural-600 text-xs py-4 text-center italic">No structures found</div>
        )}
        {!isLoading && filtered.map((s) => (
          <TreeNode
            key={s.allen_id}
            structure={s}
            depth={0}
            ontologyMapping={ontologyMapping}
            countMap={countMap}
            onRegionSelect={onRegionSelect}
          />
        ))}
      </div>

      {/* Footer note */}
      <div className="text-xs text-neural-700 border-t border-neural-800 pt-2 font-mono">
        Mouse CCF level 5 · Click acronym to view datasets
      </div>
    </div>
  )
}
