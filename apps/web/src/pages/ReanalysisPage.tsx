import { FormEvent, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getReanalysisPlan, type ReanalysisCandidate } from '../api/runtime'

function scorePercent(value: number) {
  return `${Math.round(value * 100)}%`
}

function feasibilityClass(candidate: ReanalysisCandidate) {
  if (candidate.feasibility_status === 'supported_by_metadata') return 'text-accent-emerald'
  if (candidate.feasibility_status === 'conditional_missing_signals') return 'text-amber-300'
  return 'text-red-300'
}

function CandidateCard({ candidate }: { candidate: ReanalysisCandidate }) {
  return (
    <article className="card">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex flex-wrap gap-2 items-center mb-1">
            <h3 className="text-lg font-semibold">{candidate.method_label}</h3>
            <span className="font-mono text-[11px] text-neural-500 border border-neural-800 rounded px-1.5 py-0.5">
              {candidate.analysis_family}
            </span>
          </div>
          <p className="text-sm text-neural-400 max-w-3xl">{candidate.rationale}</p>
        </div>
        <div className="text-right flex-shrink-0">
          <div className="font-mono text-xl text-accent-cyan">{scorePercent(candidate.priority_score)}</div>
          <div className="text-[11px] text-neural-600">priority</div>
        </div>
      </div>

      <div className="grid sm:grid-cols-3 gap-3 mt-4 text-xs">
        <div className="border border-neural-800 rounded p-2">
          <div className="text-neural-600 mb-1">feasibility</div>
          <div className={`font-mono ${feasibilityClass(candidate)}`}>
            {scorePercent(candidate.feasibility_score)} · {candidate.feasibility_status.replace(/_/g, ' ')}
          </div>
        </div>
        <div className="border border-neural-800 rounded p-2">
          <div className="text-neural-600 mb-1">novelty state</div>
          <div className={candidate.novelty_status === 'existing_use_evidence' ? 'text-neural-300' : 'text-amber-300'}>
            {candidate.novelty_status.replace(/_/g, ' ')}
          </div>
        </div>
        <div className="border border-neural-800 rounded p-2">
          <div className="text-neural-600 mb-1">evidence</div>
          <div className="text-neural-300">{candidate.evidence.length} records · review required</div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <div>
          <h4 className="text-xs font-medium text-neural-300 mb-2">Required data signals</h4>
          <div className="flex flex-wrap gap-1.5">
            {candidate.present_required_signals.map((signal) => (
              <span key={signal} className="text-xs text-accent-emerald border border-accent-emerald/20 rounded px-2 py-1">
                ✓ {signal}
              </span>
            ))}
            {candidate.missing_required_signals.map((signal) => (
              <span key={signal} className="text-xs text-red-300 border border-red-500/20 rounded px-2 py-1">
                ? {signal}
              </span>
            ))}
          </div>
        </div>
        {candidate.computes.length > 0 && (
          <div>
            <h4 className="text-xs font-medium text-neural-300 mb-2">Expected outputs</h4>
            <div className="flex flex-wrap gap-1.5">
              {candidate.computes.map((value) => (
                <span key={value} className="text-xs text-neural-400 border border-neural-800 rounded px-2 py-1">
                  {value.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <details className="mt-4 border-t border-neural-800 pt-3">
        <summary className="cursor-pointer text-xs text-accent-cyan">Inspect assumptions, limitations, and evidence</summary>
        <div className="grid lg:grid-cols-3 gap-5 mt-4 text-xs">
          <div>
            <h4 className="font-medium text-neural-300 mb-2">Assumptions</h4>
            {candidate.assumptions.length > 0 ? (
              <ul className="space-y-1.5 text-neural-500 list-disc pl-4">
                {candidate.assumptions.map((item) => <li key={item}>{item}</li>)}
              </ul>
            ) : <p className="text-neural-600">No structured assumptions recorded.</p>}
          </div>
          <div>
            <h4 className="font-medium text-neural-300 mb-2">Limitations</h4>
            {candidate.limitations.length > 0 ? (
              <ul className="space-y-1.5 text-neural-500 list-disc pl-4">
                {candidate.limitations.map((item) => <li key={item}>{item}</li>)}
              </ul>
            ) : <p className="text-neural-600">No structured limitations recorded.</p>}
          </div>
          <div>
            <h4 className="font-medium text-neural-300 mb-2">Evidence trail</h4>
            <div className="space-y-2">
              {candidate.evidence.map((evidence, index) => (
                <div key={`${evidence.kind}-${evidence.source_id}-${index}`} className="border border-neural-800 rounded p-2">
                  <div className="font-mono text-[10px] text-neural-600 mb-1">
                    {evidence.evidence_tier} · {evidence.kind}
                  </div>
                  <div className="text-neural-400">{evidence.summary}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </details>
    </article>
  )
}

export function ReanalysisPage() {
  const { id } = useParams<{ id?: string }>()
  const navigate = useNavigate()
  const [datasetId, setDatasetId] = useState(id ? decodeURIComponent(id) : '')

  useEffect(() => {
    if (id) setDatasetId(decodeURIComponent(id))
  }, [id])

  const plan = useQuery({
    queryKey: ['reanalysis-plan', id],
    queryFn: () => getReanalysisPlan(decodeURIComponent(id!), 12),
    enabled: Boolean(id),
    retry: false,
  })

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const value = datasetId.trim()
    if (value) navigate(`/reanalysis/${encodeURIComponent(value)}`)
  }

  return (
    <div className="max-w-5xl mx-auto px-6 lg:px-8 py-10 space-y-6">
      <section>
        <p className="font-mono text-xs text-accent-cyan mb-2">REANALYSIS WORKBENCH</p>
        <h1 className="text-3xl font-semibold">What else can this dataset support?</h1>
        <p className="text-neural-400 mt-2 max-w-3xl">
          Neural Search separates feasibility from novelty, then combines data requirements,
          method assumptions, existing paper-method evidence, related-dataset precedents, and
          literature findings into hypotheses for scientific review.
        </p>
      </section>

      <form onSubmit={submit} className="card flex flex-col sm:flex-row gap-3">
        <input
          value={datasetId}
          onChange={(event) => setDatasetId(event.target.value)}
          placeholder="Dataset ID, e.g. DEMO_VISUAL_DECISION_NEUROPIXELS"
          className="flex-1 bg-neural-900 border border-neural-700 rounded px-3 py-2 text-sm text-neural-100 outline-none focus:border-accent-cyan"
        />
        <button type="submit" className="px-4 py-2 rounded bg-accent-cyan/10 border border-accent-cyan/30 text-accent-cyan hover:text-white">
          Build plan
        </button>
      </form>

      {plan.isLoading && <div className="text-neural-500">Building evidence-aware plan…</div>}
      {plan.error && (
        <div className="card border-red-500/20 text-red-300 text-sm">
          {plan.error instanceof Error ? plan.error.message : 'Could not build a reanalysis plan.'}
        </div>
      )}

      {plan.data && (
        <>
          <section className="card">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold">{plan.data.dataset_title}</h2>
                <div className="font-mono text-xs text-neural-600 mt-1">{plan.data.dataset_id}</div>
              </div>
              <Link to={`/datasets/${encodeURIComponent(plan.data.dataset_id)}`} className="text-xs text-accent-cyan hover:text-white">
                Open dataset card →
              </Link>
            </div>
            <div className="flex flex-wrap gap-2 mt-4">
              {plan.data.matched_data_forms.map((item) => (
                <span key={item} className="text-xs border border-neural-800 rounded px-2 py-1 text-neural-400">
                  {item.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
            <div className="grid sm:grid-cols-3 gap-2 mt-4 text-xs">
              {Object.entries(plan.data.evidence_capabilities).map(([key, available]) => (
                <div key={key} className="border border-neural-800 rounded p-2">
                  <span className={available ? 'text-accent-emerald' : 'text-neural-600'}>{available ? '●' : '○'}</span>{' '}
                  <span className="text-neural-400">{key.replace(/_/g, ' ')}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-neural-600 mt-4">Corpus: {plan.data.corpus_source.replace(/_/g, ' ')}</p>
          </section>

          {plan.data.warnings.length > 0 && (
            <section className="card border-amber-500/20">
              <h2 className="text-sm font-medium text-amber-300 mb-2">Interpretation guardrails</h2>
              <ul className="text-xs text-neural-500 space-y-1.5 list-disc pl-4">
                {plan.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}
              </ul>
              <p className="text-xs text-neural-600 mt-3">{plan.data.evidence_policy}</p>
            </section>
          )}

          <section className="space-y-3">
            <div className="flex items-baseline justify-between gap-3">
              <h2 className="text-xl font-semibold">Candidate analyses</h2>
              <span className="font-mono text-xs text-neural-600">{plan.data.candidates.length} ranked hypotheses</span>
            </div>
            {plan.data.candidates.length > 0 ? (
              plan.data.candidates.map((candidate) => (
                <CandidateCard key={`${candidate.analysis_family}-${candidate.method_id}`} candidate={candidate} />
              ))
            ) : (
              <div className="card text-neural-500 text-sm">No defensible method candidates could be constructed from current metadata.</div>
            )}
          </section>
        </>
      )}
    </div>
  )
}
