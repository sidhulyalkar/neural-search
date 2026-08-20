import { useQuery } from '@tanstack/react-query'
import { getBundleState, getRuntimeStatus } from '../api/runtime'

function statusClass(available: boolean) {
  return available ? 'text-accent-emerald' : 'text-neural-600'
}

export function SystemPage() {
  const runtime = useQuery({
    queryKey: ['runtime-status'],
    queryFn: () => getRuntimeStatus(),
    staleTime: 15_000,
  })
  const bundles = useQuery({
    queryKey: ['runtime-bundles'],
    queryFn: getBundleState,
    staleTime: 15_000,
  })

  if (runtime.isLoading) {
    return <div className="max-w-5xl mx-auto px-6 py-10 text-neural-500">Loading runtime contract…</div>
  }
  if (runtime.error || !runtime.data) {
    return <div className="max-w-5xl mx-auto px-6 py-10 text-red-300">Runtime status is unavailable.</div>
  }

  const state = runtime.data
  return (
    <div className="max-w-5xl mx-auto px-6 lg:px-8 py-10 space-y-8">
      <section>
        <p className="font-mono text-xs text-accent-cyan mb-2">SYSTEM CONTRACT</p>
        <div className="flex flex-wrap items-baseline gap-3">
          <h1 className="text-3xl font-semibold">{state.profile.name}</h1>
          <span className={`font-mono text-sm ${state.health === 'ready' ? 'text-accent-emerald' : state.health === 'degraded' ? 'text-amber-300' : 'text-red-300'}`}>
            {state.health}
          </span>
        </div>
        <p className="text-neural-400 mt-2 max-w-3xl">{state.profile.description}</p>
      </section>

      <section className="card">
        <div className="flex items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-lg font-semibold">Scientific capabilities</h2>
            <p className="text-xs text-neural-500 mt-1">Availability comes from the same artifact registry used by the CLI and API.</p>
          </div>
          <span className="font-mono text-sm text-neural-400">
            {state.capabilities.available}/{state.capabilities.total} available
          </span>
        </div>
        <div className="grid sm:grid-cols-2 gap-2">
          {state.capabilities.capabilities.map((item) => (
            <div key={item.capability} className="border border-neural-800 rounded p-3 bg-neural-900/40">
              <div className="flex items-center justify-between gap-3">
                <span className="font-mono text-xs text-neural-300">{item.capability}</span>
                <span className={`text-xs ${statusClass(item.available)}`}>
                  {item.available ? 'available' : item.state.replace(/_/g, ' ')}
                </span>
              </div>
              <p className="text-xs text-neural-500 mt-2">{item.description}</p>
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-neural-600 mt-2 font-mono">
                {item.version && <span>v {item.version}</span>}
                <span>{item.resolution_source.replace(/_/g, ' ')}</span>
                {item.lineage_id && <span title={item.lineage_id}>lineage tracked</span>}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="grid md:grid-cols-2 gap-4">
        <div className="card">
          <h2 className="text-lg font-semibold mb-3">Compatibility</h2>
          <p className="font-mono text-sm text-neural-300 mb-2">{state.compatibility.state}</p>
          {state.compatibility.incompatible.length > 0 ? (
            <div className="space-y-2">
              {state.compatibility.incompatible.map((item) => (
                <div key={item.artifact_id} className="text-xs text-red-300 border border-red-500/20 rounded p-2">
                  {item.artifact_id}: {(item.issues || []).join(', ')}
                </div>
              ))}
            </div>
          ) : state.compatibility.unknown.length > 0 ? (
            <div className="space-y-2">
              {state.compatibility.unknown.slice(0, 8).map((item, index) => (
                <div key={`${item.artifact_id}-${index}`} className="text-xs text-amber-300/80">
                  {item.artifact_id}: {item.reason || 'lineage not locally verifiable'}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-accent-emerald">All tracked relationships are compatible.</p>
          )}
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold mb-3">Pinned bundle integrity</h2>
          {bundles.isLoading ? (
            <p className="text-xs text-neural-600">Checking local bundle lock…</p>
          ) : bundles.data ? (
            <>
              <p className={`font-mono text-sm mb-2 ${bundles.data.verification.valid ? 'text-accent-emerald' : 'text-red-300'}`}>
                {bundles.data.verification.valid ? 'unchanged since verification' : 'verification required'}
              </p>
              <p className="text-xs text-neural-500 break-all">{bundles.data.lock.lock_path}</p>
              <p className="text-xs text-neural-600 mt-2">
                {Object.keys(bundles.data.lock.bundles || {}).length} pinned bundles · {bundles.data.available_bundles.length} published releases
              </p>
              <p className="text-[11px] text-neural-700 mt-3">{bundles.data.verification_note}</p>
            </>
          ) : (
            <p className="text-xs text-neural-600">Bundle state is unavailable.</p>
          )}
        </div>
      </section>

      {state.recommended_missing.length > 0 && (
        <section className="card border-amber-500/20">
          <h2 className="text-lg font-semibold mb-2">Optional capability gaps</h2>
          <p className="text-xs text-neural-500 mb-3">
            These do not invalidate the active profile, but they reduce search, evidence, or reanalysis depth.
          </p>
          <div className="flex flex-wrap gap-2">
            {state.recommended_missing.map((item) => (
              <span key={item} className="font-mono text-xs text-amber-300/80 border border-amber-500/20 rounded px-2 py-1">
                {item}
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
