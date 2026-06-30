import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { atlasApi, coverageApi, type RegionDetail, type RegionDataset } from '../../api/coverage'

// ── Helpers ────────────────────────────────────────────────────────────────────

function CoverageBar({ n, max }: { n: number; max: number }) {
  const pct = max > 0 ? Math.min((n / max) * 100, 100) : 0
  const color =
    n === 0 ? '#334155' :
    n < 5 ? '#991b1b' :
    n < 30 ? '#c2410c' :
    n < 100 ? '#a16207' :
    '#047857'
  return (
    <div className="h-1 bg-neural-800 rounded-full overflow-hidden w-full">
      <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  )
}

function TopicBadge({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full border"
      style={{ borderColor: `${color}60`, backgroundColor: `${color}18`, color }}
    >
      {label}
    </span>
  )
}

function Pill({ label, dimmed = false }: { label: string; dimmed?: boolean }) {
  return (
    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${
      dimmed
        ? 'bg-neural-900 border-neural-800 text-neural-600'
        : 'bg-neural-800 border-neural-700 text-neural-400'
    }`}>
      {label}
    </span>
  )
}

function RegionChip({
  id,
  label,
  dimmed = false,
  onClick,
}: {
  id: string
  label: string
  dimmed?: boolean
  onClick?: (id: string, label: string) => void
}) {
  const cls = `text-[10px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
    dimmed
      ? 'bg-neural-900 border-neural-800 text-neural-600 cursor-default'
      : 'bg-neural-900 border-neural-700 text-neural-400 hover:border-accent-cyan/60 hover:text-neural-200 cursor-pointer'
  }`
  if (onClick && !dimmed) {
    return (
      <button type="button" className={cls} onClick={() => onClick(id, label)}>
        {label}
      </button>
    )
  }
  return <span className={cls}>{label}</span>
}

// ── Section wrapper ────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <span className="block text-[9px] uppercase tracking-widest text-neural-600 font-mono">{title}</span>
      {children}
    </div>
  )
}

// ── Hierarchy breadcrumb chain ─────────────────────────────────────────────────

function HierarchyChain({
  detail,
  onRegionClick,
}: {
  detail: RegionDetail
  onRegionClick: (id: string, label: string) => void
}) {
  const allNodes = [
    ...detail.parents,
    { id: detail.id, label: detail.label, isCurrent: true },
    ...detail.children.slice(0, 3),
  ] as Array<{ id: string; label: string; isCurrent?: boolean }>

  return (
    <div className="flex flex-wrap items-center gap-1">
      {allNodes.map((node, idx) => (
        <span key={node.id} className="flex items-center gap-1">
          {idx > 0 && (
            <span className="text-neural-700 text-[10px]">›</span>
          )}
          {node.isCurrent ? (
            <span className="text-[10px] font-mono font-semibold text-accent-cyan px-1.5 py-0.5 bg-accent-cyan/10 border border-accent-cyan/30 rounded">
              {node.label}
            </span>
          ) : (
            <button
              type="button"
              className="text-[10px] font-mono text-neural-400 hover:text-neural-200 transition-colors"
              onClick={() => onRegionClick(node.id, node.label)}
            >
              {node.label}
            </button>
          )}
        </span>
      ))}
      {detail.children.length > 3 && (
        <span className="text-[10px] text-neural-700 font-mono">+{detail.children.length - 3} more</span>
      )}
    </div>
  )
}

// ── Cross-reference row ────────────────────────────────────────────────────────

function CrossRefs({ detail }: { detail: RegionDetail }) {
  const refs = [
    detail.allen_structure && {
      key: 'Allen CCF',
      value: `${detail.allen_structure.acronym} (#${detail.allen_structure.allen_id})`,
      color: detail.allen_structure.color_hex ? `#${detail.allen_structure.color_hex}` : '#22d3ee',
      href: `https://atlas.brain-map.org/atlas?atlas=1&plate=100883869#atlas=1&plate=100883869&resolution=10&x=5700&y=4000&zoom=-3&structure=${detail.allen_structure.allen_id}`,
    },
    detail.atlas_refs.uberon && {
      key: 'UBERON',
      value: detail.atlas_refs.uberon,
      color: '#8b5cf6',
      href: `https://www.ebi.ac.uk/ols4/ontologies/uberon/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F${detail.atlas_refs.uberon.replace(':', '_')}`,
    },
    detail.atlas_refs.allen_human && {
      key: 'Allen Human',
      value: `#${detail.atlas_refs.allen_human}`,
      color: '#10b981',
      href: `https://atlas.brain-map.org/atlas?atlas=10#structure=${detail.atlas_refs.allen_human}`,
    },
    detail.atlas_refs.waxholm_rat && {
      key: 'Waxholm Rat',
      value: `#${detail.atlas_refs.waxholm_rat}`,
      color: '#f59e0b',
      href: undefined,
    },
  ].filter(Boolean) as Array<{ key: string; value: string; color: string; href?: string }>

  if (refs.length === 0) return null

  return (
    <Section title="Atlas Cross-References">
      <div className="flex flex-col gap-1">
        {refs.map((ref) => (
          <div key={ref.key} className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: ref.color }} />
            <span className="text-[10px] font-mono text-neural-600 w-20 flex-shrink-0">{ref.key}</span>
            {ref.href ? (
              <a
                href={ref.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[10px] font-mono text-neural-400 hover:text-accent-cyan transition-colors truncate"
              >
                {ref.value}
              </a>
            ) : (
              <span className="text-[10px] font-mono text-neural-400 truncate">{ref.value}</span>
            )}
          </div>
        ))}
      </div>
    </Section>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export function RegionDetailPanel({
  regionId,
  regionLabel,
  onRegionClick,
}: {
  regionId: string
  regionLabel: string
  onRegionClick: (id: string, label: string) => void
}) {
  const { data: detail, isLoading: detailLoading } = useQuery({
    queryKey: ['region-detail', regionId],
    queryFn: () => atlasApi.regionDetail(regionId),
    staleTime: 300_000,
    retry: false,
  })

  const { data: datasetsResp, isLoading: dsLoading } = useQuery({
    queryKey: ['region-datasets', regionId],
    queryFn: () => coverageApi.regionDatasets(regionId, { limit: 15 }),
    enabled: !!regionId,
    staleTime: 60_000,
  })

  const nDatasets = datasetsResp?.count ?? 0
  const maxDatasets = 400

  if (detailLoading) {
    return (
      <div className="space-y-3 animate-pulse">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-3 bg-neural-800 rounded" style={{ width: `${60 + i * 10}%` }} />
        ))}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div>
        <h2 className="font-mono text-sm text-white leading-tight">{regionLabel}</h2>
        {detail?.aliases && detail.aliases.length > 0 && (
          <p className="text-[10px] text-neural-600 mt-0.5 italic">
            aka {detail.aliases.slice(0, 3).join(', ')}
            {detail.aliases.length > 3 && ` +${detail.aliases.length - 3}`}
          </p>
        )}
      </div>

      {/* Coverage bar */}
      <Section title={`Dataset Coverage — ${nDatasets} datasets`}>
        <CoverageBar n={nDatasets} max={maxDatasets} />
        {nDatasets === 0 && (
          <p className="text-[10px] text-red-400 font-mono mt-1">
            No datasets — research gap
          </p>
        )}
        {nDatasets > 0 && nDatasets < 5 && (
          <p className="text-[10px] text-orange-400 font-mono mt-1">
            Sparse coverage — {nDatasets} dataset{nDatasets !== 1 ? 's' : ''}
          </p>
        )}
      </Section>

      {/* Hierarchy */}
      {detail && (detail.parents.length > 0 || detail.children.length > 0) && (
        <Section title="Ontology Position">
          <HierarchyChain detail={detail} onRegionClick={onRegionClick} />
        </Section>
      )}

      {/* Sibling regions */}
      {detail && detail.siblings.length > 0 && (
        <Section title="Sibling Regions">
          <div className="flex flex-wrap gap-1">
            {detail.siblings.map((s) => (
              <RegionChip key={s.id} id={s.id} label={s.label} onClick={onRegionClick} />
            ))}
          </div>
        </Section>
      )}

      {/* Connected topics */}
      {detail && detail.connected_topics.length > 0 && (
        <Section title="Research Topics">
          <div className="flex flex-wrap gap-1">
            {detail.connected_topics.map((t) => (
              <TopicBadge key={t.id} label={t.label} color={t.color} />
            ))}
          </div>
          {detail.connected_topics[0]?.description && (
            <p className="text-[10px] text-neural-600 mt-1 leading-snug italic">
              {detail.connected_topics[0].description}
            </p>
          )}
        </Section>
      )}

      {/* Cross-references */}
      {detail && <CrossRefs detail={detail} />}

      {/* Datasets list */}
      {nDatasets > 0 && (
        <Section title={`Datasets (${nDatasets})`}>
          {dsLoading && (
            <div className="text-neural-600 text-[10px] font-mono">Loading…</div>
          )}
          <div className="flex flex-col gap-1.5 mt-1">
            {datasetsResp?.datasets.map((ds: RegionDataset) => (
              <Link
                key={ds.dataset_id}
                to={`/datasets/${encodeURIComponent(ds.dataset_id)}`}
                className="block bg-neural-900/60 border border-neural-800 rounded-lg px-2.5 py-2 hover:border-accent-cyan/50 transition-colors group"
              >
                <p className="text-[11px] text-neural-200 font-medium leading-snug line-clamp-2 group-hover:text-white transition-colors">
                  {ds.title}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <Pill label={ds.source} />
                  <span className="text-[9px] text-neural-700 font-mono">
                    {((ds.confidence ?? 0) * 100).toFixed(0)}% conf
                  </span>
                  {ds.access_tier && (
                    <Pill label={ds.access_tier} dimmed />
                  )}
                </div>
              </Link>
            ))}
          </div>
          {nDatasets > 15 && (
            <p className="text-[10px] text-neural-600 font-mono mt-1">
              Showing 15 of {nDatasets} — use search for full list
            </p>
          )}
        </Section>
      )}

      {/* Empty state for no ontology data */}
      {!detail && !detailLoading && nDatasets === 0 && (
        <div className="text-center py-6">
          <p className="text-neural-600 text-xs italic">
            No ontology entry or datasets for this region.
          </p>
        </div>
      )}
    </div>
  )
}
