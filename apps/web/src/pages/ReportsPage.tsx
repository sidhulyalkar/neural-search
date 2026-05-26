import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getCompilationReport } from '../api/search'
import { SpinnerIcon, ChartBarIcon, BeakerIcon, ChevronRightIcon } from '../components/Icons'

interface BarChartProps {
  data: Record<string, number>
  maxBars?: number
  colorClass?: string
}

function BarChart({ data, maxBars = 10, colorClass = 'bg-accent-cyan' }: BarChartProps) {
  const entries = Object.entries(data)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxBars)

  const maxValue = Math.max(...entries.map(([, v]) => v), 1)

  return (
    <div className="space-y-2">
      {entries.map(([label, count]) => (
        <div key={label} className="flex items-center gap-3">
          <div className="w-32 text-sm text-neural-400 truncate" title={label}>
            {label.replace(/_/g, ' ')}
          </div>
          <div className="flex-1 h-5 bg-neural-800 rounded overflow-hidden">
            <div
              className={`h-full ${colorClass} transition-all duration-500`}
              style={{ width: `${(count / maxValue) * 100}%` }}
            />
          </div>
          <div className="w-10 text-sm text-neural-300 text-right">{count}</div>
        </div>
      ))}
    </div>
  )
}

export function ReportsPage() {
  const { data: report, isLoading, error } = useQuery({
    queryKey: ['compilation-report'],
    queryFn: getCompilationReport,
  })

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" aria-live="polite" aria-busy="true">
        <div className="flex items-center gap-3 text-neural-400 mb-6">
          <SpinnerIcon className="w-5 h-5 text-accent-cyan" />
          <span>Loading corpus coverage, readiness, and missing metadata report...</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 animate-pulse">
          {[0, 1, 2, 3].map((item) => (
            <div key={item} className="card h-24" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-pulse">
          <div className="card h-64" />
          <div className="card h-64" />
        </div>
      </div>
    )
  }

  if (error || !report) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <p className="text-red-300 font-medium mb-2">Failed to load compilation report</p>
        <p className="text-sm text-neural-400 mb-4">
          {error instanceof Error
            ? error.message
            : 'The API could not return corpus report data.'}
        </p>
        <Link to="/" className="text-accent-cyan hover:underline">
          Return to search
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-neural-100 mb-2 flex items-center gap-3">
          <ChartBarIcon className="w-8 h-8 text-accent-cyan" />
          Dataset Compilation Report
        </h1>
        <p className="text-neural-400">
          Overview of all indexed neural datasets and their distribution across categories
        </p>
        {report.generated_at && (
          <p className="text-sm text-neural-500 mt-2">
            Generated: {new Date(report.generated_at).toLocaleString()}
          </p>
        )}
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="card text-center">
          <div className="text-3xl font-bold text-accent-cyan">{report.total_datasets}</div>
          <div className="text-sm text-neural-400">Total Datasets</div>
        </div>
        <div className="card text-center">
          <div className="text-3xl font-bold text-accent-violet">
            {Object.keys(report.datasets_by_source || {}).length}
          </div>
          <div className="text-sm text-neural-400">Sources</div>
        </div>
        <div className="card text-center">
          <div className="text-3xl font-bold text-accent-emerald">
            {Object.keys(report.datasets_by_task || {}).length}
          </div>
          <div className="text-sm text-neural-400">Task Types</div>
        </div>
        <div className="card text-center">
          <div className="text-3xl font-bold text-accent-amber">
            {Object.keys(report.datasets_by_modality || {}).length}
          </div>
          <div className="text-sm text-neural-400">Modalities</div>
        </div>
      </div>

      {report.qa_review_counts && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="card text-center">
            <div className="text-3xl font-bold text-accent-cyan">{report.qa_review_counts.reviewed || 0}</div>
            <div className="text-sm text-neural-400">Reviewed</div>
          </div>
          <div className="card text-center">
            <div className="text-3xl font-bold text-accent-emerald">{report.qa_review_counts.trusted || 0}</div>
            <div className="text-sm text-neural-400">Trusted</div>
          </div>
          <div className="card text-center">
            <div className="text-3xl font-bold text-red-300">{report.qa_review_counts.rejected || 0}</div>
            <div className="text-sm text-neural-400">Rejected</div>
          </div>
        </div>
      )}

      {/* Charts grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Datasets by Source */}
        {report.datasets_by_source && Object.keys(report.datasets_by_source).length > 0 && (
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Datasets by Source</h2>
            <BarChart data={report.datasets_by_source} colorClass="bg-accent-cyan" />
          </section>
        )}

        {/* Datasets by Task */}
        {report.datasets_by_task && Object.keys(report.datasets_by_task).length > 0 && (
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Datasets by Task</h2>
            <BarChart data={report.datasets_by_task} colorClass="bg-accent-violet" />
          </section>
        )}

        {/* Datasets by Modality */}
        {report.datasets_by_modality && Object.keys(report.datasets_by_modality).length > 0 && (
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Datasets by Modality</h2>
            <BarChart data={report.datasets_by_modality} colorClass="bg-accent-emerald" />
          </section>
        )}

        {/* Datasets by Species */}
        {report.datasets_by_species && Object.keys(report.datasets_by_species).length > 0 && (
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Datasets by Species</h2>
            <BarChart data={report.datasets_by_species} colorClass="bg-accent-amber" />
          </section>
        )}

        {/* Datasets by Brain Region */}
        {report.datasets_by_brain_region && Object.keys(report.datasets_by_brain_region).length > 0 && (
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Datasets by Brain Region</h2>
            <BarChart data={report.datasets_by_brain_region} colorClass="bg-neural-400" />
          </section>
        )}

        {/* Datasets by Data Standard */}
        {report.datasets_by_data_standard && Object.keys(report.datasets_by_data_standard).length > 0 && (
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Datasets by Data Standard</h2>
            <BarChart data={report.datasets_by_data_standard} colorClass="bg-accent-cyan" />
          </section>
        )}

        {report.common_missing_metadata && Object.keys(report.common_missing_metadata).length > 0 && (
          <section className="card">
            <h2 className="text-lg font-semibold mb-4">Common Missing Metadata</h2>
            <BarChart data={report.common_missing_metadata} colorClass="bg-red-400" />
          </section>
        )}
      </div>

      {report.total_datasets === 0 && (
        <div className="card text-center py-12 mb-8">
          <p className="text-neural-200 font-medium mb-2">No datasets have been compiled yet</p>
          <p className="text-sm text-neural-500">
            Run <code className="text-accent-cyan">make demo</code> or <code className="text-accent-cyan">make reports</code> to generate demo corpus artifacts.
          </p>
        </div>
      )}

      {/* Top Demo-Ready Datasets */}
      {report.top_demo_ready && report.top_demo_ready.length > 0 && (
        <section className="card mb-8">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BeakerIcon className="w-5 h-5 text-accent-cyan" />
            Top Datasets Ready for Demo
          </h2>
          <div className="space-y-3">
            {report.top_demo_ready.map((dataset, index) => (
              <div
                key={dataset.dataset_id}
                className="flex items-center justify-between py-3 px-4 bg-neural-800/50 rounded hover:bg-neural-800 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <span className="text-xl font-bold text-neural-500 w-8">
                    #{index + 1}
                  </span>
                  <div>
                    <Link
                      to={`/datasets/${dataset.dataset_id}`}
                      className="text-neural-200 font-medium hover:text-accent-cyan transition-colors flex items-center gap-1"
                    >
                      {dataset.title}
                      <ChevronRightIcon className="w-4 h-4" />
                    </Link>
                    <span className="text-sm text-neural-500">
                      {dataset.source.toUpperCase()} · {dataset.qa_status.replace(/_/g, ' ')}
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-accent-cyan">
                    {Math.round(dataset.score)}
                  </div>
                  <div className="text-xs text-neural-500">demo score</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Top Analysis-Ready Datasets */}
      {report.top_analysis_ready && report.top_analysis_ready.length > 0 && (
        <section className="card">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BeakerIcon className="w-5 h-5 text-accent-emerald" />
            Top Analysis-Ready Datasets
          </h2>
          <div className="space-y-3">
            {report.top_analysis_ready.map((dataset, index) => (
              <div
                key={dataset.dataset_id}
                className="flex items-center justify-between py-3 px-4 bg-neural-800/50 rounded hover:bg-neural-800 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <span className="text-xl font-bold text-neural-500 w-8">
                    #{index + 1}
                  </span>
                  <div>
                    <Link
                      to={`/datasets/${dataset.dataset_id}`}
                      className="text-neural-200 font-medium hover:text-accent-cyan transition-colors flex items-center gap-1"
                    >
                      {dataset.title}
                      <ChevronRightIcon className="w-4 h-4" />
                    </Link>
                    <span className="text-sm text-neural-500">
                      {dataset.source.toUpperCase()}
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-accent-emerald">
                    {Math.round(dataset.score)}
                  </div>
                  <div className="text-xs text-neural-500">readiness</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Info box */}
      <div className="mt-8 card bg-neural-800/50">
        <h3 className="font-medium text-neural-200 mb-2">About This Report</h3>
        <p className="text-sm text-neural-400">
          This compilation report provides an overview of all indexed neural datasets.
          It shows the distribution of datasets across different sources, tasks, modalities,
          species, and brain regions. The analysis readiness score indicates how well-prepared
          each dataset is for downstream analysis.
        </p>
      </div>
    </div>
  )
}
