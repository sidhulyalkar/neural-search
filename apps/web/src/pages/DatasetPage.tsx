import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { getDataset, generateNotebook } from '../api/search'
import { SpinnerIcon, ExternalLinkIcon, CodeIcon, DocumentIcon, CheckCircleIcon, WarningIcon } from '../components/Icons'

export function DatasetPage() {
  const { id } = useParams<{ id: string }>()

  const { data: card, isLoading, error } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => getDataset(id!),
    enabled: !!id,
  })

  const notebookMutation = useMutation({
    mutationFn: () => generateNotebook(id!),
    onSuccess: (blob) => {
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${id}_starter.ipynb`
      a.click()
      URL.revokeObjectURL(url)
    },
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <SpinnerIcon className="w-8 h-8 text-accent-cyan" />
      </div>
    )
  }

  if (error || !card) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <p className="text-red-400 mb-4">Failed to load dataset</p>
        <Link to="/" className="text-accent-cyan hover:underline">
          Return to search
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Breadcrumb */}
      <nav className="text-sm text-neural-500 mb-6">
        <Link to="/" className="hover:text-neural-300">Search</Link>
        <span className="mx-2">/</span>
        <span className="text-neural-300">{card.title}</span>
      </nav>

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-3">
          <span className={`badge ${card.source === 'dandi' ? 'badge-cyan' : 'badge-violet'}`}>
            {card.source.toUpperCase()}
          </span>
          {card.data_standard && (
            <span className="badge badge-emerald">{card.data_standard.toUpperCase()}</span>
          )}
        </div>
        <h1 className="text-3xl font-bold text-neural-100 mb-4">{card.title}</h1>
        <p className="text-neural-400">{card.summary}</p>
      </div>

      {/* Actions */}
      <div className="flex gap-4 mb-8">
        {card.url && (
          <a
            href={card.url}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary flex items-center gap-2"
          >
            <ExternalLinkIcon className="w-4 h-4" />
            View Source
          </a>
        )}
        <button
          onClick={() => notebookMutation.mutate()}
          disabled={notebookMutation.isPending}
          className="btn-primary flex items-center gap-2"
        >
          <CodeIcon className="w-4 h-4" />
          {notebookMutation.isPending ? 'Generating...' : 'Generate Notebook'}
        </button>
      </div>

      {/* Content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Metadata */}
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Overview</h2>
            <dl className="space-y-3">
              {card.species.length > 0 && (
                <div>
                  <dt className="text-sm text-neural-500">Species</dt>
                  <dd className="flex gap-2 mt-1">
                    {card.species.map((s) => (
                      <span key={s} className="badge bg-neural-700">{s}</span>
                    ))}
                  </dd>
                </div>
              )}
              {card.tasks.length > 0 && (
                <div>
                  <dt className="text-sm text-neural-500">Tasks</dt>
                  <dd className="flex flex-wrap gap-2 mt-1">
                    {card.tasks.map((t) => (
                      <span key={t} className="badge bg-neural-700">{t}</span>
                    ))}
                  </dd>
                </div>
              )}
              {card.modalities.length > 0 && (
                <div>
                  <dt className="text-sm text-neural-500">Modalities</dt>
                  <dd className="flex flex-wrap gap-2 mt-1">
                    {card.modalities.map((m) => (
                      <span key={m} className="badge bg-neural-700">{m}</span>
                    ))}
                  </dd>
                </div>
              )}
              {card.brain_regions.length > 0 && (
                <div>
                  <dt className="text-sm text-neural-500">Brain Regions</dt>
                  <dd className="flex flex-wrap gap-2 mt-1">
                    {card.brain_regions.map((r) => (
                      <span key={r} className="badge bg-neural-700">{r}</span>
                    ))}
                  </dd>
                </div>
              )}
              {card.doi && (
                <div>
                  <dt className="text-sm text-neural-500">DOI</dt>
                  <dd className="font-mono text-sm text-neural-300 mt-1">{card.doi}</dd>
                </div>
              )}
            </dl>
          </section>

          {/* Suggested analyses */}
          {card.readiness?.suggested_analyses && card.readiness.suggested_analyses.length > 0 && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Suggested Analyses</h2>
              <ul className="space-y-2">
                {card.readiness.suggested_analyses.map((analysis) => (
                  <li key={analysis} className="flex items-center gap-2 text-neural-300">
                    <DocumentIcon className="w-4 h-4 text-accent-cyan" />
                    {analysis.replace(/_/g, ' ')}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Readiness score */}
          {card.readiness && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Analysis Readiness</h2>
              <div className="flex items-center gap-4 mb-4">
                <div className="text-4xl font-bold text-accent-cyan">
                  {Math.round(card.readiness.score * 100)}
                </div>
                <div className="text-sm text-neural-400">/ 100</div>
              </div>

              {card.readiness.strengths.length > 0 && (
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-neural-300 mb-2">Strengths</h3>
                  <ul className="space-y-1">
                    {card.readiness.strengths.map((s) => (
                      <li key={s} className="flex items-start gap-2 text-sm text-neural-400">
                        <CheckCircleIcon className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {card.readiness.limitations.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-neural-300 mb-2">Limitations</h3>
                  <ul className="space-y-1">
                    {card.readiness.limitations.map((l) => (
                      <li key={l} className="flex items-start gap-2 text-sm text-neural-400">
                        <WarningIcon className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                        {l}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </section>
          )}

          {/* Related papers */}
          {card.related_papers.length > 0 && (
            <section className="card">
              <h2 className="text-lg font-semibold mb-4">Related Papers</h2>
              <ul className="space-y-2">
                {card.related_papers.map((paper) => (
                  <li key={paper} className="text-sm text-neural-400 hover:text-neural-300">
                    {paper}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}
