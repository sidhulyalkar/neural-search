import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getDemoReport } from '../api/search'

const SUB_QUERIES = [
  {
    id: 'SQ1',
    query: 'prefrontal cortex hippocampus interaction working memory',
    intent: 'cross_dataset_comparison',
    regions: ['prefrontal_cortex', 'hippocampus'],
    task: 'working_memory',
  },
  {
    id: 'SQ2',
    query: 'dopamine reward prediction error striatum',
    intent: 'meta_analysis',
    regions: ['striatum'],
    task: 'reward_learning',
  },
  {
    id: 'SQ3',
    query: 'motor cortex adaptation learning plasticity',
    intent: 'method_transfer',
    regions: ['motor_cortex'],
    task: 'motor_task',
  },
  {
    id: 'SQ4',
    query: 'cross-species decision making flexible behavior reversal learning',
    intent: 'cross_dataset_comparison',
    regions: ['prefrontal_cortex', 'striatum'],
    task: 'decision_making',
  },
  {
    id: 'SQ5',
    query: 'population dynamics prefrontal cortex latent space manifold',
    intent: 'method_transfer',
    regions: ['prefrontal_cortex'],
    task: 'any',
  },
]

const INTENT_COLORS: Record<string, string> = {
  cross_dataset_comparison: 'text-accent-cyan',
  meta_analysis: 'text-accent-violet',
  method_transfer: 'text-accent-emerald',
}

const ROLE_COLORS: Record<string, string> = {
  anchor: 'text-accent-cyan border-accent-cyan/30 bg-accent-cyan/5',
  high_relevance: 'text-neural-300 border-neural-700 bg-neural-900',
  replication: 'text-accent-violet border-accent-violet/30 bg-accent-violet/5',
  cross_species_comparator: 'text-accent-emerald border-accent-emerald/30 bg-accent-emerald/5',
  methodological_complement: 'text-amber-400 border-amber-400/30 bg-amber-400/5',
  perturbation_causal: 'text-rose-400 border-rose-400/30 bg-rose-400/5',
  behavior_rich: 'text-orange-400 border-orange-400/30 bg-orange-400/5',
  population_dynamics: 'text-blue-400 border-blue-400/30 bg-blue-400/5',
  imaging_ephys_bridge: 'text-purple-400 border-purple-400/30 bg-purple-400/5',
}

interface DemoReport {
  run_at: string
  query: string
  n_candidates: number
  set_coverage: {
    total_score: number
    coverage_bonus: number
    complementarity_bonus: number
  }
  role_assignments: Array<{
    dataset_id: string
    role: string
    title: string
    evidence: string
  }>
  hard_criteria: {
    all_datasets_have_role: boolean
    zero_hard_negative_violations: boolean
    anchor_assigned: boolean
  }
  coverage_criteria: {
    n_distinct_roles: number
    coverage_bonus: number
    complementarity_bonus: number
  }
  all_hard_criteria_pass: boolean
}

function Stage({
  num,
  label,
  sub,
  done,
}: {
  num: number
  label: string
  sub?: string
  done?: boolean
}) {
  return (
    <div className="flex flex-col items-center text-center">
      <div
        className={`w-8 h-8 rounded-full border flex items-center justify-center text-xs font-mono mb-2 transition-colors ${
          done
            ? 'border-accent-cyan/50 bg-accent-cyan/10 text-accent-cyan'
            : 'border-neural-700 bg-neural-900 text-neural-500'
        }`}
      >
        {done ? '✓' : num}
      </div>
      <div className={`text-xs font-medium ${done ? 'text-neural-200' : 'text-neural-500'}`}>{label}</div>
      {sub && <div className="text-xs text-neural-600 mt-0.5">{sub}</div>}
    </div>
  )
}

export function DemoPage() {
  const { data, isLoading, error } = useQuery<DemoReport>({
    queryKey: ['demo-report'],
    queryFn: getDemoReport,
    retry: false,
  })

  const allPass = data?.all_hard_criteria_pass ?? false

  return (
    <div className="max-w-5xl mx-auto px-6 lg:px-8 py-16">
      {/* Header */}
      <div className="mb-16">
        <div className="mb-3">
          <span className="font-mono text-xs text-neural-600 tracking-widest uppercase">
            Killer Demo · Multi-Dataset Retrieval
          </span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-extralight tracking-tight text-neural-100 mb-6 leading-tight">
          Cognitive Control<br />
          <span className="text-neural-500">Query Pipeline</span>
        </h1>
        <p className="text-neural-500 text-sm max-w-xl leading-relaxed">
          A single complex query decomposed into five sub-queries, retrieving across 835 datasets,
          scored for set coverage, and assigned roles for multi-dataset analysis.
        </p>
      </div>

      {/* The query */}
      <section className="mb-16">
        <p className="text-xs text-neural-600 uppercase tracking-widest mb-4">The Query</p>
        <blockquote className="border-l-2 border-accent-cyan/40 pl-6 py-2">
          <p className="text-lg sm:text-xl text-neural-200 font-light leading-relaxed">
            "Map the neural circuit mechanisms underlying flexible cognitive control — integrating
            datasets spanning prefrontal-hippocampal interactions, dopaminergic reward modulation,
            motor adaptation, and cross-species learning-dependent plasticity — to identify
            convergent computational mechanisms."
          </p>
        </blockquote>
      </section>

      {/* Pipeline stages */}
      <section className="mb-16">
        <p className="text-xs text-neural-600 uppercase tracking-widest mb-6">5-Stage Pipeline</p>
        <div className="flex items-start justify-between relative">
          {/* Connector line */}
          <div className="absolute top-4 left-0 right-0 h-px bg-neural-800 z-0 mx-16" />
          {[
            { label: 'Decompose', sub: '5 sub-queries' },
            { label: 'Retrieve', sub: `${data?.n_candidates ?? '…'} candidates` },
            { label: 'Score', sub: `${data ? (data.set_coverage.coverage_bonus * 100).toFixed(0) : '…'}% coverage` },
            { label: 'Assign', sub: `${data?.role_assignments.length ?? '…'} datasets` },
            { label: 'Validate', sub: allPass ? 'PASS' : (data ? 'FAIL' : '…') },
          ].map((s, i) => (
            <div key={s.label} className="relative z-10 flex-1">
              <Stage num={i + 1} label={s.label} sub={s.sub} done={!!data} />
            </div>
          ))}
        </div>
      </section>

      {/* Sub-queries */}
      <section className="mb-16">
        <p className="text-xs text-neural-600 uppercase tracking-widest mb-4">
          Stage 1 — Sub-Query Decomposition
        </p>
        <div className="space-y-px">
          {SUB_QUERIES.map((sq) => (
            <div
              key={sq.id}
              className="flex items-start gap-4 py-3 border-b border-neural-800/40 last:border-0"
            >
              <span className="font-mono text-xs text-neural-600 w-8 flex-shrink-0 pt-0.5">{sq.id}</span>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-neural-200 mb-1">{sq.query}</p>
                <div className="flex flex-wrap gap-2">
                  <span className={`text-xs font-mono ${INTENT_COLORS[sq.intent] || 'text-neural-500'}`}>
                    {sq.intent.replace(/_/g, ' ')}
                  </span>
                  <span className="text-xs text-neural-600">·</span>
                  <span className="text-xs text-neural-600">{sq.regions.map(r => r.replace(/_/g, ' ')).join(', ')}</span>
                  {sq.task !== 'any' && (
                    <>
                      <span className="text-xs text-neural-600">·</span>
                      <span className="text-xs text-neural-600">{sq.task.replace(/_/g, ' ')}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Coverage results */}
      {data && (
        <section className="mb-16">
          <p className="text-xs text-neural-600 uppercase tracking-widest mb-4">
            Stage 3 — Set Coverage Scoring
          </p>
          <div className="grid grid-cols-3 gap-px bg-neural-800/30 border border-neural-800/30 rounded-lg overflow-hidden">
            <div className="bg-neural-950 px-6 py-5">
              <div className="text-3xl font-extralight text-neural-100 tabular-nums mb-1">
                {data.n_candidates}
              </div>
              <div className="text-xs text-neural-600">candidates retrieved</div>
            </div>
            <div className="bg-neural-950 px-6 py-5">
              <div className="text-3xl font-extralight text-accent-cyan tabular-nums mb-1">
                {(data.set_coverage.coverage_bonus * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-neural-600">coverage bonus</div>
            </div>
            <div className="bg-neural-950 px-6 py-5">
              <div className="text-3xl font-extralight text-neural-100 tabular-nums mb-1">
                {data.set_coverage.total_score.toFixed(2)}
              </div>
              <div className="text-xs text-neural-600">total score</div>
            </div>
          </div>
        </section>
      )}

      {/* Role assignments */}
      {data && data.role_assignments.length > 0 && (
        <section className="mb-16">
          <p className="text-xs text-neural-600 uppercase tracking-widest mb-4">
            Stage 4 — Role Assignment · {data.role_assignments.length} datasets
          </p>
          <div className="space-y-px">
            {data.role_assignments.map((ra) => (
              <div
                key={ra.dataset_id}
                className="flex items-start gap-4 py-3 border-b border-neural-800/40 last:border-0 group"
              >
                <div className="flex-shrink-0 pt-0.5">
                  <span
                    className={`inline-block text-xs font-medium border rounded px-2 py-0.5 ${
                      ROLE_COLORS[ra.role] || 'text-neural-400 border-neural-700 bg-neural-900'
                    }`}
                  >
                    {ra.role.replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="flex-1 min-w-0">
                  <Link
                    to={`/datasets/${ra.dataset_id}`}
                    className="text-sm text-neural-200 hover:text-white transition-colors line-clamp-1 block"
                  >
                    {ra.title || ra.dataset_id}
                  </Link>
                  {ra.evidence && (
                    <p className="text-xs text-neural-600 mt-0.5 line-clamp-1">{ra.evidence}</p>
                  )}
                </div>
                <Link
                  to={`/datasets/${ra.dataset_id}`}
                  className="text-xs text-neural-700 group-hover:text-neural-400 transition-colors flex-shrink-0 pt-0.5"
                >
                  →
                </Link>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Success metrics */}
      {data && (
        <section className="mb-16">
          <p className="text-xs text-neural-600 uppercase tracking-widest mb-4">
            Stage 5 — Validation
          </p>
          <div className="space-y-2">
            {Object.entries(data.hard_criteria).map(([key, passed]) => (
              <div key={key} className="flex items-center gap-3">
                <span className={`font-mono text-sm ${passed ? 'text-accent-cyan' : 'text-red-400'}`}>
                  {passed ? '✓' : '✗'}
                </span>
                <span className="text-sm text-neural-400">{key.replace(/_/g, ' ')}</span>
              </div>
            ))}
          </div>

          <div className="mt-6 inline-flex items-center gap-3">
            <div
              className={`text-sm font-medium px-4 py-2 rounded border ${
                allPass
                  ? 'border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan'
                  : 'border-red-500/30 bg-red-500/10 text-red-400'
              }`}
            >
              {allPass ? 'All criteria pass' : 'Check hard criteria'}
            </div>
            {data.run_at && (
              <span className="text-xs text-neural-700">
                {new Date(data.run_at).toLocaleDateString()}
              </span>
            )}
          </div>
        </section>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="py-16 text-center">
          <span className="inline-flex items-center gap-3 text-neural-500 text-sm">
            <span className="w-4 h-4 border-2 border-neural-700 border-t-accent-cyan rounded-full animate-spin" />
            Loading demo report…
          </span>
        </div>
      )}

      {/* Error / not found */}
      {error && (
        <div className="py-10 border border-neural-800/50 rounded-lg px-6">
          <p className="text-neural-300 text-sm mb-2">Demo report not available</p>
          <p className="text-neural-600 text-xs mb-3">
            Generate it by running the killer demo script.
          </p>
          <code className="text-xs text-neural-400 bg-neural-900 border border-neural-800 rounded px-3 py-2 block font-mono">
            python scripts/run_killer_demo.py
          </code>
        </div>
      )}

      {/* Footer CTA */}
      <div className="border-t border-neural-800/40 pt-10 flex flex-wrap gap-6">
        <Link to="/" className="text-sm text-neural-500 hover:text-accent-cyan transition-colors">
          ← Try a search
        </Link>
        <Link to="/graph" className="text-sm text-neural-500 hover:text-accent-cyan transition-colors">
          Corpus map →
        </Link>
      </div>
    </div>
  )
}
