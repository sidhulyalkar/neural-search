import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getDemoReport } from '../api/search'
import {
  BeakerIcon,
  ChartBarIcon,
  CheckCircleIcon,
  CodeIcon,
  DocumentIcon,
  SearchIcon,
  WarningIcon,
} from '../components/Icons'

interface DemoSubQuery {
  id: string
  query: string
  intent: string
  regions?: string[]
  brain_regions?: string[]
  species_constraint?: string[]
  task?: string
  task_family?: string
}

interface RoleAssignment {
  dataset_id: string
  role: string
  title: string
  evidence: string
}

interface DemoReport {
  run_at: string
  query: string
  sub_queries?: DemoSubQuery[]
  n_candidates: number
  set_coverage: {
    total_score: number
    coverage_bonus: number
    complementarity_bonus: number
  }
  role_assignments: RoleAssignment[]
  hard_criteria: Record<string, boolean>
  coverage_criteria: {
    n_distinct_roles: number
    coverage_bonus: number
    complementarity_bonus: number
  }
  all_hard_criteria_pass: boolean
}

const FALLBACK_SUB_QUERIES: DemoSubQuery[] = [
  {
    id: 'SQ1',
    query: 'prefrontal cortex hippocampus interaction working memory',
    intent: 'cross_dataset_comparison',
    brain_regions: ['prefrontal_cortex', 'hippocampus'],
    task_family: 'working_memory',
  },
  {
    id: 'SQ2',
    query: 'dopamine reward prediction error striatum',
    intent: 'meta_analysis',
    brain_regions: ['striatum'],
    task_family: 'reward_learning',
  },
  {
    id: 'SQ3',
    query: 'motor cortex adaptation learning plasticity',
    intent: 'method_transfer',
    brain_regions: ['motor_cortex'],
    task_family: 'motor_task',
  },
  {
    id: 'SQ4',
    query: 'cross-species decision making flexible behavior reversal learning',
    intent: 'cross_dataset_comparison',
    brain_regions: ['prefrontal_cortex', 'striatum'],
    task_family: 'decision_making',
  },
  {
    id: 'SQ5',
    query: 'population dynamics prefrontal cortex latent space manifold',
    intent: 'method_transfer',
    brain_regions: ['prefrontal_cortex'],
    task_family: 'any',
  },
]

const DEMO_SCENARIOS = [
  {
    label: 'Reversal learning',
    query: 'Find reversal learning datasets with reward omission and trial outcomes',
    focus: 'Task labels, reward events, trial outcomes',
  },
  {
    label: 'Calcium imaging',
    query: 'Go/NoGo task with calcium imaging in mPFC and lick events',
    focus: 'Modality, behavior, brain region normalization',
  },
  {
    label: 'Human BCI',
    query: 'Human ECoG or iEEG reaching data for BCI decoding - multiple subjects',
    focus: 'Human invasive recordings and reuse readiness',
  },
  {
    label: 'Open data audit',
    query: 'DANDI datasets with NWB format, calcium imaging, and published analysis notebooks',
    focus: 'Provenance, data standard, notebook affordances',
  },
]

const SHOWCASE_ARTIFACTS = [
  {
    title: 'Experiment search',
    detail: 'Natural language plus structured filters route into ontology, metadata, semantic, and provenance signals.',
    href: '/',
    icon: SearchIcon,
  },
  {
    title: 'Dataset cards',
    detail: 'Each result can open a reuse card with readiness, warnings, linked literature, and notebook export.',
    href: '/search?q=Find%20reversal%20learning%20datasets%20with%20reward%20omission',
    icon: DocumentIcon,
  },
  {
    title: 'Corpus graph',
    detail: 'The map exposes source, species, modality, region, and task coverage as searchable concept rings.',
    href: '/graph',
    icon: BeakerIcon,
  },
  {
    title: 'Evaluation dashboard',
    detail: 'Benchmark runs validate whether retrieval recovers expected scientific labels and reusable datasets.',
    href: '/evaluation',
    icon: ChartBarIcon,
  },
]

const INTENT_COLORS: Record<string, string> = {
  cross_dataset_comparison: 'text-accent-cyan border-accent-cyan/30 bg-accent-cyan/10',
  meta_analysis: 'text-accent-violet border-accent-violet/30 bg-accent-violet/10',
  method_transfer: 'text-accent-emerald border-accent-emerald/30 bg-accent-emerald/10',
}

const ROLE_COLORS: Record<string, string> = {
  anchor: 'text-accent-cyan border-accent-cyan/30 bg-accent-cyan/10',
  high_relevance: 'text-neural-300 border-neural-700 bg-neural-900',
  replication: 'text-accent-violet border-accent-violet/30 bg-accent-violet/10',
  cross_species_comparator: 'text-accent-emerald border-accent-emerald/30 bg-accent-emerald/10',
  methodological_complement: 'text-amber-400 border-amber-400/30 bg-amber-400/10',
  perturbation_causal: 'text-rose-400 border-rose-400/30 bg-rose-400/10',
  behavior_rich: 'text-orange-400 border-orange-400/30 bg-orange-400/10',
  population_dynamics: 'text-blue-400 border-blue-400/30 bg-blue-400/10',
  imaging_ephys_bridge: 'text-purple-400 border-purple-400/30 bg-purple-400/10',
}

function formatPercent(value: number | undefined, digits = 0) {
  if (value === undefined || Number.isNaN(value)) return '--'
  const percent = value <= 1 ? value * 100 : value
  return `${percent.toFixed(digits)}%`
}

function formatDate(value: string | undefined) {
  if (!value) return 'No run loaded'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

function readable(value: string) {
  return value.replace(/_/g, ' ')
}

function searchHref(query: string) {
  return `/search?q=${encodeURIComponent(query)}`
}

function groupRoleAssignments(assignments: RoleAssignment[]) {
  return assignments.reduce<Record<string, RoleAssignment[]>>((groups, assignment) => {
    const role = assignment.role || 'unclassified'
    groups[role] = groups[role] || []
    groups[role].push(assignment)
    return groups
  }, {})
}

function StatusPill({ pass }: { pass: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded border px-3 py-1 text-xs font-medium ${
        pass
          ? 'border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan'
          : 'border-red-500/30 bg-red-500/10 text-red-300'
      }`}
    >
      {pass ? <CheckCircleIcon className="h-3.5 w-3.5" /> : <WarningIcon className="h-3.5 w-3.5" />}
      {pass ? 'Validated' : 'Needs review'}
    </span>
  )
}

function MetricCard({
  label,
  value,
  sub,
  tone = 'neutral',
}: {
  label: string
  value: string | number
  sub: string
  tone?: 'neutral' | 'cyan' | 'emerald'
}) {
  const valueColor = tone === 'cyan' ? 'text-accent-cyan' : tone === 'emerald' ? 'text-accent-emerald' : 'text-neural-100'
  return (
    <div className="bg-neural-950 px-5 py-4">
      <div className={`text-3xl font-extralight tabular-nums ${valueColor}`}>{value}</div>
      <div className="mt-1 text-sm text-neural-300">{label}</div>
      <div className="mt-0.5 text-xs text-neural-600">{sub}</div>
    </div>
  )
}

function ProgressBar({
  label,
  value,
  detail,
}: {
  label: string
  value: number
  detail: string
}) {
  const clamped = Math.max(0, Math.min(value, 1))
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-4">
        <span className="text-sm text-neural-300">{label}</span>
        <span className="font-mono text-xs text-neural-500">{formatPercent(clamped)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded bg-neural-900">
        <div className="h-full rounded bg-accent-cyan" style={{ width: `${clamped * 100}%` }} />
      </div>
      <p className="mt-1 text-xs text-neural-600">{detail}</p>
    </div>
  )
}

function SubQueryCard({ subQuery }: { subQuery: DemoSubQuery }) {
  const regions = subQuery.brain_regions || subQuery.regions || []
  const task = subQuery.task_family || subQuery.task
  return (
    <Link
      to={searchHref(subQuery.query)}
      className="block rounded-lg border border-neural-800/70 bg-neural-950 p-4 transition-colors hover:border-accent-cyan/50"
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <span className="font-mono text-xs text-neural-500">{subQuery.id}</span>
        <span
          className={`rounded border px-2 py-0.5 text-xs ${
            INTENT_COLORS[subQuery.intent] || 'border-neural-700 bg-neural-900 text-neural-500'
          }`}
        >
          {readable(subQuery.intent)}
        </span>
      </div>
      <p className="text-sm leading-relaxed text-neural-200">{subQuery.query}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {task && (
          <span className="rounded bg-neural-900 px-2 py-0.5 text-xs text-neural-500">
            {readable(task)}
          </span>
        )}
        {regions.slice(0, 3).map((region) => (
          <span key={region} className="rounded bg-neural-900 px-2 py-0.5 text-xs text-neural-500">
            {readable(region)}
          </span>
        ))}
      </div>
    </Link>
  )
}

function ValidationRow({
  label,
  passed,
  detail,
}: {
  label: string
  passed: boolean
  detail: string
}) {
  return (
    <div className="flex items-start gap-3 border-b border-neural-800/40 py-3 last:border-0">
      <span className={`mt-0.5 font-mono text-sm ${passed ? 'text-accent-cyan' : 'text-red-400'}`}>
        {passed ? 'PASS' : 'WARN'}
      </span>
      <div>
        <div className="text-sm text-neural-200">{label}</div>
        <div className="mt-0.5 text-xs text-neural-600">{detail}</div>
      </div>
    </div>
  )
}

export function DemoPage() {
  const { data, isLoading, error } = useQuery<DemoReport>({
    queryKey: ['demo-report'],
    queryFn: getDemoReport,
    retry: false,
  })

  const subQueries = data?.sub_queries?.length ? data.sub_queries : FALLBACK_SUB_QUERIES
  const roleGroups = useMemo(
    () => groupRoleAssignments(data?.role_assignments || []),
    [data?.role_assignments],
  )
  const roleNames = Object.keys(roleGroups).sort((a, b) => roleGroups[b].length - roleGroups[a].length)
  const allPass = data?.all_hard_criteria_pass ?? false
  const validationRows = data
    ? [
        ...Object.entries(data.hard_criteria).map(([key, passed]) => ({
          label: readable(key),
          passed,
          detail:
            key === 'all_datasets_have_role'
              ? 'Every selected dataset has an explicit analytic role.'
              : key === 'zero_hard_negative_violations'
                ? 'The run did not surface hard-negative rule violations.'
                : 'At least one anchor dataset was assigned for the showcase set.',
        })),
        {
          label: 'role diversity',
          passed: data.coverage_criteria.n_distinct_roles >= 2,
          detail: `${data.coverage_criteria.n_distinct_roles} distinct roles assigned across selected datasets.`,
        },
        {
          label: 'coverage bonus',
          passed: data.coverage_criteria.coverage_bonus >= 0.75,
          detail: `${formatPercent(data.coverage_criteria.coverage_bonus)} coverage bonus across decomposed sub-queries.`,
        },
      ]
    : []

  return (
    <div className="mx-auto max-w-6xl px-6 py-12 lg:px-8">
      <section className="mb-12 grid gap-8 lg:grid-cols-[1.15fr_0.85fr] lg:items-end">
        <div>
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <span className="font-mono text-xs uppercase tracking-widest text-neural-600">
              Full demo showcase
            </span>
            {data && <StatusPill pass={allPass} />}
          </div>
          <h1 className="text-4xl font-extralight tracking-tight text-neural-100 sm:text-5xl">
            Evaluate, validate, and present the retrieval demo.
          </h1>
          <p className="mt-5 max-w-2xl text-sm leading-relaxed text-neural-500">
            This page turns the killer demo report into a guided product walkthrough: decomposed
            scientific intent, candidate retrieval, set coverage scoring, role assignment, validation
            checks, and links into the live app surfaces.
          </p>
        </div>

        <div className="rounded-lg border border-neural-800/70 bg-neural-950 p-5">
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <p className="section-label">Latest report</p>
              <p className="mt-2 text-sm text-neural-300">{formatDate(data?.run_at)}</p>
            </div>
            <CodeIcon className="h-5 w-5 text-neural-600" />
          </div>
          <code className="block rounded border border-neural-800 bg-neural-900 px-3 py-2 font-mono text-xs text-neural-400">
            python scripts/run_killer_demo.py
          </code>
          <p className="mt-3 text-xs leading-relaxed text-neural-600">
            Regenerate the JSON report, then refresh this page to validate the current demo run.
          </p>
        </div>
      </section>

      {isLoading && (
        <div className="mb-12 rounded-lg border border-neural-800/60 px-6 py-10 text-center">
          <span className="inline-flex items-center gap-3 text-sm text-neural-500">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-neural-700 border-t-accent-cyan" />
            Loading demo report...
          </span>
        </div>
      )}

      {error && (
        <div className="mb-12 rounded-lg border border-amber-500/30 bg-amber-500/5 px-6 py-5">
          <div className="flex items-start gap-3">
            <WarningIcon className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-400" />
            <div>
              <p className="text-sm font-medium text-amber-200">Demo report not available</p>
              <p className="mt-1 text-xs leading-relaxed text-neural-500">
                The showcase still works with built-in demo scenarios. Generate the latest validation
                artifact with <code className="font-mono text-neural-300">python scripts/run_killer_demo.py</code>.
              </p>
            </div>
          </div>
        </div>
      )}

      <section className="mb-12 grid gap-px overflow-hidden rounded-lg border border-neural-800/30 bg-neural-800/30 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Candidates"
          value={data?.n_candidates ?? '--'}
          sub="retrieved across sub-queries"
        />
        <MetricCard
          label="Coverage"
          value={formatPercent(data?.set_coverage.coverage_bonus)}
          sub="sub-query coverage bonus"
          tone="cyan"
        />
        <MetricCard
          label="Roles"
          value={data?.coverage_criteria.n_distinct_roles ?? '--'}
          sub="distinct dataset roles"
          tone="emerald"
        />
        <MetricCard
          label="Validation"
          value={data ? (allPass ? 'PASS' : 'WARN') : '--'}
          sub="hard criteria gate"
          tone={allPass ? 'cyan' : 'neutral'}
        />
      </section>

      <section className="mb-12 rounded-lg border border-neural-800/70 bg-neural-950 p-6">
        <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div>
            <p className="section-label">Showcase query</p>
            <blockquote className="mt-3 max-w-4xl border-l-2 border-accent-cyan/40 pl-5 text-lg font-light leading-relaxed text-neural-200">
              {data?.query ||
                'Map the neural circuit mechanisms underlying flexible cognitive control across circuit, reward, motor, and cross-species learning datasets.'}
            </blockquote>
          </div>
          <Link
            to={searchHref(data?.query || DEMO_SCENARIOS[0].query)}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-accent-cyan px-4 py-2 text-sm font-medium text-neural-950 transition-colors hover:bg-accent-cyan/90"
          >
            <SearchIcon className="h-4 w-4" />
            Open search
          </Link>
        </div>

        {data && (
          <div className="grid gap-5 border-t border-neural-800/50 pt-6 md:grid-cols-3">
            <ProgressBar
              label="Total set score"
              value={data.set_coverage.total_score}
              detail="Aggregate strength of the selected multi-dataset set."
            />
            <ProgressBar
              label="Coverage bonus"
              value={data.set_coverage.coverage_bonus}
              detail="How well the retrieved set spans the decomposed scientific intent."
            />
            <ProgressBar
              label="Complementarity"
              value={data.set_coverage.complementarity_bonus}
              detail="Extra reward for datasets that add non-redundant evidence."
            />
          </div>
        )}
      </section>

      <section className="mb-12">
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <p className="section-label">Guided demo scenarios</p>
            <h2 className="mt-2 text-2xl font-light text-neural-100">Launch useful searches</h2>
          </div>
          <Link to="/reports" className="text-sm text-neural-500 transition-colors hover:text-accent-cyan">
            Corpus reports &rarr;
          </Link>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {DEMO_SCENARIOS.map((scenario) => (
            <Link
              key={scenario.label}
              to={searchHref(scenario.query)}
              className="rounded-lg border border-neural-800/70 bg-neural-950 p-4 transition-colors hover:border-accent-cyan/50"
            >
              <div className="mb-2 flex items-center justify-between gap-4">
                <span className="text-sm font-medium text-neural-200">{scenario.label}</span>
                <span className="text-neural-700">&rarr;</span>
              </div>
              <p className="text-sm leading-relaxed text-neural-500">{scenario.query}</p>
              <p className="mt-3 text-xs text-neural-600">{scenario.focus}</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="mb-12">
        <p className="section-label mb-4">Stage 1 - sub-query decomposition</p>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {subQueries.map((subQuery) => (
            <SubQueryCard key={subQuery.id} subQuery={subQuery} />
          ))}
        </div>
      </section>

      {data && (
        <section className="mb-12 grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
          <div className="rounded-lg border border-neural-800/70 bg-neural-950 p-6">
            <div className="mb-5 flex items-start justify-between gap-4">
              <div>
                <p className="section-label">Stage 5 - validation</p>
                <h2 className="mt-2 text-2xl font-light text-neural-100">
                  {allPass ? 'Demo gates pass' : 'Review demo gates'}
                </h2>
              </div>
              <StatusPill pass={allPass} />
            </div>
            <div>
              {validationRows.map((row) => (
                <ValidationRow
                  key={row.label}
                  label={row.label}
                  passed={row.passed}
                  detail={row.detail}
                />
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-neural-800/70 bg-neural-950 p-6">
            <p className="section-label">Stages 2-4 - result roles</p>
            <h2 className="mt-2 text-2xl font-light text-neural-100">Showcase dataset set</h2>
            <p className="mt-2 text-sm leading-relaxed text-neural-600">
              {data.role_assignments.length} datasets are assigned analytic roles so the demo reads
              as a reusable set, not a plain ranked list.
            </p>
            <div className="mt-5 space-y-5">
              {roleNames.map((role) => (
                <div key={role}>
                  <div className="mb-2 flex items-center gap-2">
                    <span
                      className={`rounded border px-2 py-0.5 text-xs ${
                        ROLE_COLORS[role] || 'border-neural-700 bg-neural-900 text-neural-400'
                      }`}
                    >
                      {readable(role)}
                    </span>
                    <span className="text-xs text-neural-600">{roleGroups[role].length} datasets</span>
                  </div>
                  <div className="space-y-px">
                    {roleGroups[role].slice(0, 5).map((assignment) => (
                      <Link
                        key={assignment.dataset_id}
                        to={`/datasets/${encodeURIComponent(assignment.dataset_id)}`}
                        className="group block border-b border-neural-800/40 py-2 last:border-0"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0">
                            <p className="truncate text-sm text-neural-200 group-hover:text-white">
                              {assignment.title || assignment.dataset_id}
                            </p>
                            <p className="mt-0.5 truncate text-xs text-neural-600">
                              {assignment.evidence}
                            </p>
                          </div>
                          <span className="flex-shrink-0 text-neural-700 group-hover:text-accent-cyan">
                            &rarr;
                          </span>
                        </div>
                      </Link>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      <section className="mb-12">
        <p className="section-label mb-4">Full-form product surfaces</p>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
          {SHOWCASE_ARTIFACTS.map((artifact) => {
            const Icon = artifact.icon
            return (
              <Link
                key={artifact.title}
                to={artifact.href}
                className="rounded-lg border border-neural-800/70 bg-neural-950 p-4 transition-colors hover:border-accent-cyan/50"
              >
                <Icon className="mb-4 h-5 w-5 text-accent-cyan" />
                <h3 className="text-sm font-medium text-neural-200">{artifact.title}</h3>
                <p className="mt-2 text-xs leading-relaxed text-neural-600">{artifact.detail}</p>
              </Link>
            )
          })}
        </div>
      </section>

      <div className="flex flex-wrap gap-6 border-t border-neural-800/40 pt-8">
        <Link to="/" className="text-sm text-neural-500 transition-colors hover:text-accent-cyan">
          Try a search &rarr;
        </Link>
        <Link to="/graph" className="text-sm text-neural-500 transition-colors hover:text-accent-cyan">
          Corpus map &rarr;
        </Link>
        <Link to="/evaluation" className="text-sm text-neural-500 transition-colors hover:text-accent-cyan">
          Evaluation dashboard &rarr;
        </Link>
      </div>
    </div>
  )
}
