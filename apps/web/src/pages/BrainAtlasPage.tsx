import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { coverageApi, type RegionCount, type RegionDataset } from '../api/coverage'

// ── Anatomical grouping ───────────────────────────────────────────────────────
// Maps display group label → region_id values (from data/ontology/brain_regions.yaml)
const ATLAS_GROUPS: Record<string, string[]> = {
  'Visual System': [
    'visual_cortex', 'v1', 'v2', 'v4', 'area_mt', 'inferior_temporal_cortex',
    'lateral_geniculate',
  ],
  'Frontal / Motor': [
    'prefrontal_cortex', 'dlpfc', 'ofc', 'vlpfc', 'anterior_cingulate_cortex',
    'motor_cortex', 'm1', 'premotor_cortex', 'supplementary_motor_area',
    'orbitofrontal_cortex',
  ],
  'Somatosensory': [
    'somatosensory_cortex', 's1', 's2', 'barrel_cortex',
  ],
  'Parietal / Temporal': [
    'parietal_cortex', 'posterior_parietal_cortex', 'temporal_cortex',
    'auditory_cortex', 'inferior_parietal_lobule',
  ],
  'Hippocampal Fm.': [
    'hippocampus', 'ca1', 'ca3', 'dentate_gyrus', 'entorhinal_cortex',
    'subiculum', 'parahippocampal_cortex',
  ],
  'Cingulate / Insula': [
    'cingulate_cortex', 'insular_cortex', 'retrosplenial_cortex',
    'medial_prefrontal_cortex',
  ],
  'Basal Ganglia': [
    'striatum', 'caudate_nucleus', 'putamen', 'nucleus_accumbens',
    'globus_pallidus', 'subthalamic_nucleus', 'basal_ganglia',
  ],
  'Thalamus': [
    'thalamus', 'mediodorsal_thalamus', 'pulvinar', 'medial_geniculate',
    'ventral_posterior_thalamus',
  ],
  'Amygdala': [
    'amygdala', 'lateral_amygdala', 'basal_amygdala', 'central_amygdala',
    'bed_nucleus_stria_terminalis',
  ],
  'Hypothalamus': [
    'hypothalamus', 'lateral_hypothalamus', 'paraventricular_nucleus',
  ],
  'Brainstem': [
    'midbrain', 'superior_colliculus', 'inferior_colliculus',
    'substantia_nigra', 'ventral_tegmental_area', 'periaqueductal_gray',
    'pons', 'locus_coeruleus', 'raphe_nucleus', 'medulla',
  ],
  'Cerebellum': [
    'cerebellum', 'cerebellar_cortex', 'deep_cerebellar_nuclei',
  ],
}

// ── Colour helpers ─────────────────────────────────────────────────────────────
function tileClasses(n: number, selected: boolean): string {
  const base = 'relative border rounded px-2 py-1.5 text-left transition-all cursor-pointer'
  const ring = selected ? ' ring-2 ring-accent-cyan ring-offset-1 ring-offset-neural-950' : ''

  if (n === 0) return `${base}${ring} bg-neural-900 border-neural-800 text-neural-600 cursor-default`
  if (n < 5)   return `${base}${ring} bg-red-950 border-red-800/50 text-red-300 hover:bg-red-900/60`
  if (n < 30)  return `${base}${ring} bg-orange-950 border-orange-700/50 text-orange-300 hover:bg-orange-900/60`
  if (n < 100) return `${base}${ring} bg-yellow-950 border-yellow-600/50 text-yellow-300 hover:bg-yellow-900/60`
  if (n < 400) return `${base}${ring} bg-emerald-950 border-emerald-700/50 text-emerald-300 hover:bg-emerald-900/60`
  return `${base}${ring} bg-emerald-900 border-emerald-500/70 text-emerald-100 hover:bg-emerald-800`
}

// ── Sub-components ─────────────────────────────────────────────────────────────
function RegionTile({
  label,
  nDatasets,
  selected,
  onClick,
}: {
  regionId: string
  label: string
  nDatasets: number
  selected: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={nDatasets > 0 ? onClick : undefined}
      title={`${label} — ${nDatasets} datasets`}
      className={tileClasses(nDatasets, selected)}
    >
      <div className="text-xs font-medium leading-tight truncate">{label}</div>
      <div className="text-xs font-mono opacity-70 mt-0.5">{nDatasets}</div>
    </button>
  )
}

function RegionPanel({ regionId, regionLabel }: { regionId: string; regionLabel: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['region-datasets', regionId],
    queryFn: () => coverageApi.regionDatasets(regionId, { limit: 30 }),
    enabled: !!regionId,
  })

  return (
    <div className="flex flex-col h-full gap-3">
      <div>
        <h2 className="font-mono text-base text-white leading-tight">{regionLabel}</h2>
        <p className="text-xs text-neural-500 mt-0.5">
          {isLoading ? '…' : `${data?.count ?? 0} datasets`}
        </p>
      </div>

      {isLoading && <div className="text-neural-500 text-sm">Loading…</div>}
      {data && data.datasets.length === 0 && (
        <div className="text-neural-600 text-sm italic">No datasets found.</div>
      )}

      <div className="flex flex-col gap-2 overflow-y-auto">
        {data?.datasets.map((ds: RegionDataset) => (
          <Link
            key={ds.dataset_id}
            to={`/datasets/${encodeURIComponent(ds.dataset_id)}`}
            className="block bg-neural-900 border border-neural-800 rounded-lg p-3 hover:border-accent-cyan/50 transition-colors"
          >
            <div className="text-sm text-neural-100 font-medium leading-snug mb-1 line-clamp-2">
              {ds.title}
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-neural-500 font-mono">{ds.source}</span>
              <span className="text-xs text-neural-700">·</span>
              <span className="text-xs text-neural-500">
                {((ds.confidence ?? 0) * 100).toFixed(0)}% conf
              </span>
              {ds.access_tier && (
                <>
                  <span className="text-xs text-neural-700">·</span>
                  <span className="text-xs text-neural-600">{ds.access_tier}</span>
                </>
              )}
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

function Legend() {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span className="text-xs text-neural-500">Coverage:</span>
      {[
        { label: '0', cls: 'bg-neural-900 border-neural-800' },
        { label: '1–4', cls: 'bg-red-950 border-red-800/50' },
        { label: '5–29', cls: 'bg-orange-950 border-orange-700/50' },
        { label: '30–99', cls: 'bg-yellow-950 border-yellow-600/50' },
        { label: '100–399', cls: 'bg-emerald-950 border-emerald-700/50' },
        { label: '400+', cls: 'bg-emerald-900 border-emerald-500/70' },
      ].map((e) => (
        <div key={e.label} className="flex items-center gap-1">
          <div className={`w-3 h-3 rounded border ${e.cls}`} />
          <span className="text-xs text-neural-500">{e.label}</span>
        </div>
      ))}
    </div>
  )
}

// ── Page ───────────────────────────────────────────────────────────────────────
export function BrainAtlasPage() {
  const [selected, setSelected] = useState<{ id: string; label: string } | null>(null)

  const { data: counts, isLoading } = useQuery({
    queryKey: ['region-counts'],
    queryFn: coverageApi.regionCounts,
  })

  const countMap = new Map<string, { n: number; label: string }>(
    (counts ?? []).map((r: RegionCount) => [r.region_id, { n: r.n_datasets, label: r.region_label }])
  )

  const coveredRegions = counts?.filter((r) => r.n_datasets > 0).length ?? 0
  const totalInAtlas = Object.values(ATLAS_GROUPS).flat().length

  function handleTileClick(id: string, label: string) {
    setSelected((prev) => (prev?.id === id ? null : { id, label }))
  }

  return (
    <div className="max-w-7xl mx-auto px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-5">
        <h1 className="font-mono text-2xl text-white mb-1">Brain Atlas</h1>
        <p className="text-neural-500 text-sm">
          Dataset coverage across brain regions. Click any region to explore its datasets.
        </p>
      </div>

      {/* Stats + legend */}
      {counts && (
        <div className="flex items-center gap-6 mb-5 flex-wrap text-sm">
          <span>
            <span className="font-mono text-white">{coveredRegions}</span>
            <span className="text-neural-500 ml-1">/ {totalInAtlas} regions covered</span>
          </span>
          <span className="text-neural-700 hidden sm:inline">·</span>
          <Legend />
        </div>
      )}

      {isLoading && <div className="text-neural-500 text-sm">Loading atlas…</div>}

      {/* Atlas grid + side panel */}
      <div className="flex gap-6 items-start">
        {/* Left: grouped region grid */}
        <div className="flex-1 min-w-0">
          {Object.entries(ATLAS_GROUPS).map(([groupLabel, regionIds]) => {
            const regions = regionIds.map((rid) => {
              const info = countMap.get(rid)
              return {
                id: rid,
                label: info?.label ?? rid.replace(/_/g, ' '),
                n: info?.n ?? 0,
              }
            })
            const groupTotal = regions.reduce((s, r) => s + r.n, 0)
            return (
              <div key={groupLabel} className="mb-5">
                <div className="flex items-baseline gap-2 mb-2">
                  <span className="text-xs font-mono text-neural-400 uppercase tracking-wider">
                    {groupLabel}
                  </span>
                  <span className="text-xs text-neural-700">{groupTotal.toLocaleString()} datasets</span>
                </div>
                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-1.5">
                  {regions.map((r) => (
                    <RegionTile
                      key={r.id}
                      regionId={r.id}
                      label={r.label}
                      nDatasets={r.n}
                      selected={selected?.id === r.id}
                      onClick={() => handleTileClick(r.id, r.label)}
                    />
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        {/* Right: selected region panel */}
        <div className="w-72 flex-shrink-0 hidden md:block">
          <div className="sticky top-24 bg-neural-900/50 border border-neural-800 rounded-xl p-4 max-h-[80vh] overflow-y-auto">
            {selected ? (
              <RegionPanel regionId={selected.id} regionLabel={selected.label} />
            ) : (
              <div className="text-center py-12">
                <div className="text-neural-600 text-sm">
                  Click a brain region to explore its datasets
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile: selected region (below grid) */}
      {selected && (
        <div className="md:hidden mt-6 bg-neural-900/50 border border-neural-800 rounded-xl p-4">
          <RegionPanel regionId={selected.id} regionLabel={selected.label} />
        </div>
      )}
    </div>
  )
}
