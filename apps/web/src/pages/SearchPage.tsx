import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getArtifactManifest } from '../api/search'

const exampleQueries = [
  'Multi-region calcium imaging of PFC, hippocampus, and striatum during working memory',
  'Cross-species comparison of decision-making: human fMRI and rodent electrophysiology',
  'Large-scale Neuropixels recordings across cortex and thalamus with behavioral readout',
  'Reversal learning datasets with reward omission, trial outcomes, and mPFC recordings',
  'Human ECoG or iEEG reaching data for BCI decoding — multiple subjects',
  'DANDI datasets with NWB format, calcium imaging, and published analysis notebooks',
  'Delay discounting with fiber photometry across multiple limbic regions',
  'OpenNeuro fMRI with task design, behavioral logfiles, and BIDS derivatives',
]

const pillars = [
  { label: 'Ontology', sub: 'tasks · behaviors · regions' },
  { label: 'Metadata', sub: 'species · modality · standards' },
  { label: 'Semantic', sub: 'dense embedding · intent matching' },
  { label: 'Provenance', sub: 'papers · QA · readiness' },
]

type EvidenceTier = 'live' | 'silver' | 'pending'

interface EvidenceStat {
  label: string
  value: string
  tier: EvidenceTier
}

function buildEvidenceStats(manifest: Awaited<ReturnType<typeof getArtifactManifest>> | undefined): EvidenceStat[] {
  if (!manifest) {
    return [
      { label: 'Dataset records', value: '7,171', tier: 'live' },
      { label: 'Linked papers', value: '168 DOI / 225 fuzzy', tier: 'live' },
      { label: 'Extracted findings', value: '~12K (operational)', tier: 'silver' },
      { label: 'Silver qrels', value: '175', tier: 'silver' },
      { label: 'Human adjudicated', value: '3 (pending)', tier: 'pending' },
      { label: 'Gold benchmark', value: 'Pending', tier: 'pending' },
    ]
  }
  const { corpus, literature, qrels } = manifest
  const doi = literature.doi_exact_links
  const fuzzy = literature.title_fuzzy_links
  const findings = literature.findings_operational_estimate
  return [
    { label: 'Dataset records', value: corpus.row_count.toLocaleString(), tier: 'live' },
    { label: 'Linked papers', value: `${doi.toLocaleString()} DOI / ${fuzzy.toLocaleString()} fuzzy`, tier: 'live' },
    {
      label: 'Extracted findings',
      value: findings > 0 ? `~${(findings / 1000).toFixed(0)}K (operational)` : 'Pending',
      tier: 'silver',
    },
    { label: 'Silver qrels', value: qrels.silver.rows.toLocaleString(), tier: 'silver' },
    {
      label: 'Human adjudicated',
      value: qrels.field_state_adjudicated.rows > 0
        ? `${qrels.field_state_adjudicated.rows} (smoke-test)`
        : '0 (pending)',
      tier: 'pending',
    },
    {
      label: 'Gold benchmark',
      value: qrels.gold.rows > 0 ? qrels.gold.rows.toLocaleString() : 'Pending qrel completion',
      tier: 'pending',
    },
  ]
}

export function SearchPage() {
  const [query, setQuery] = useState('')
  const navigate = useNavigate()

  const { data: manifest } = useQuery({
    queryKey: ['artifact-manifest'],
    queryFn: getArtifactManifest,
    staleTime: 5 * 60 * 1000,
    retry: false,
  })

  const corpusCount = manifest?.corpus.row_count.toLocaleString() ?? '7,171'
  const evidenceStats = buildEvidenceStats(manifest)

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`)
    }
  }

  const handleExampleClick = (example: string) => {
    navigate(`/search?q=${encodeURIComponent(example)}`)
  }

  return (
    <div className="max-w-5xl mx-auto px-6 lg:px-8">
      {/* Hero */}
      <div className="pt-24 pb-16">
        <div className="mb-3">
          <span className="font-mono text-xs text-neural-600 tracking-widest uppercase">
            v2.0 · {corpusCount} dataset records
          </span>
        </div>

        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extralight tracking-tight text-neural-100 leading-none mb-6">
          Neural Search
        </h1>

        <p className="text-lg sm:text-xl text-neural-400 font-light max-w-2xl leading-relaxed mb-2">
          Experiment-aware semantic search for reusable neuroscience datasets.
        </p>
        <p className="text-sm text-neural-600 max-w-xl">
          Describe task structure, recording modality, species, and analysis intent.
        </p>
      </div>

      {/* Search */}
      <form onSubmit={handleSearch} className="mb-16">
        <div className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What experiment are you trying to reuse?"
            className="flex-1 bg-neural-900 border border-neural-700 rounded-lg px-5 py-4 text-neural-100 text-base placeholder-neural-600 focus:outline-none focus:border-neural-500 focus:ring-1 focus:ring-neural-500 transition-colors"
            autoFocus
          />
          <button
            type="submit"
            className="bg-accent-cyan text-neural-950 font-medium px-6 py-4 rounded-lg hover:bg-accent-cyan/90 transition-colors text-sm whitespace-nowrap"
          >
            Search
          </button>
        </div>
      </form>

      {/* Four pillars */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-neural-800/30 border border-neural-800/30 rounded-lg overflow-hidden mb-16">
        {pillars.map(({ label, sub }) => (
          <div key={label} className="bg-neural-950 px-5 py-4">
            <div className="text-sm font-medium text-neural-200 mb-0.5">{label}</div>
            <div className="text-xs text-neural-600">{sub}</div>
          </div>
        ))}
      </div>

      {/* Evidence tier strip — live from /api/artifacts/current-manifest */}
      <div className="mb-12 border border-neural-800/40 rounded-lg px-5 py-4">
        <p className="text-xs text-neural-600 uppercase tracking-widest mb-3">Evidence Status</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {evidenceStats.map(({ label, value, tier }) => (
            <div key={label} className="flex flex-col gap-1">
              <span className="text-xs text-neural-600">{label}</span>
              <span
                className={`text-xs font-mono font-medium ${
                  tier === 'live'
                    ? 'text-accent-cyan'
                    : tier === 'silver'
                      ? 'text-amber-400'
                      : 'text-neural-600'
                }`}
              >
                {value}
              </span>
              {tier === 'silver' && (
                <span className="text-[10px] text-amber-600 uppercase tracking-wider">
                  silver · diagnostic
                </span>
              )}
              {tier === 'pending' && (
                <span className="text-[10px] text-neural-700 uppercase tracking-wider">pending</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Example queries */}
      <div className="mb-16">
        <p className="text-xs text-neural-600 uppercase tracking-widest mb-4">Example queries</p>
        <div className="space-y-1">
          {exampleQueries.map((example) => (
            <button
              key={example}
              onClick={() => handleExampleClick(example)}
              className="block w-full text-left text-sm text-neural-500 hover:text-neural-200 py-2 border-b border-neural-800/50 last:border-0 transition-colors group"
            >
              <span className="group-hover:text-accent-cyan mr-2 transition-colors">→</span>
              {example}
            </button>
          ))}
        </div>
      </div>

      {/* Bottom links */}
      <div className="flex flex-wrap gap-6 pb-16">
        <Link to="/demo" className="text-sm text-neural-500 hover:text-accent-cyan transition-colors">
          Query Pipeline Demo — cognitive control →
        </Link>
        <Link to="/graph" className="text-sm text-neural-500 hover:text-accent-cyan transition-colors">
          Corpus Map — {corpusCount} dataset records →
        </Link>
        <Link to="/ontology" className="text-sm text-neural-500 hover:text-neural-200 transition-colors">
          Browse Ontology →
        </Link>
      </div>
    </div>
  )
}
