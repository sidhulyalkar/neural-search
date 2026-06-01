import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getEvaluationReport, runEvaluation } from '../api/search'
import {
  SpinnerIcon,
  CheckCircleIcon,
  XCircleIcon,
  ChevronRightIcon,
  BeakerIcon,
} from '../components/Icons'

export function EvaluationPage() {
  const queryClient = useQueryClient()

  const { data: report, isLoading, error } = useQuery({
    queryKey: ['evaluation-report'],
    queryFn: getEvaluationReport,
  })

  const runMutation = useMutation({
    mutationFn: runEvaluation,
    onSuccess: (data) => {
      queryClient.setQueryData(['evaluation-report'], data)
    },
  })

  const avgPrecision = report?.avg_precision_at_5 ?? 0
  const avgRecall = report?.avg_label_recall_at_10 ?? 0
  const passRate = report && report.total_queries > 0 ? (report.passed_queries / report.total_queries) * 100 : 0

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-neural-100 mb-2 flex items-center gap-3">
          <BeakerIcon className="w-8 h-8 text-accent-violet" />
          Evaluation Dashboard
        </h1>
        <p className="text-neural-400">
          Benchmark search quality with predefined queries and expected results
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mb-8">
        <button
          onClick={() => runMutation.mutate()}
          disabled={runMutation.isPending}
          className="btn-primary flex items-center gap-2"
        >
          {runMutation.isPending ? (
            <>
              <SpinnerIcon className="w-4 h-4" />
              Running Benchmark...
            </>
          ) : (
            <>
              <BeakerIcon className="w-4 h-4" />
              Run Benchmark
            </>
          )}
        </button>

        {report?.timestamp && (
          <p className="text-sm text-neural-500">
            Last run: {new Date(report.timestamp).toLocaleString()}
          </p>
        )}
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="space-y-4" aria-live="polite" aria-busy="true">
          <div className="flex items-center gap-3 text-neural-400">
            <SpinnerIcon className="w-5 h-5 text-accent-cyan" />
            <span>Preparing benchmark report for the demo corpus...</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-pulse">
            {[0, 1, 2, 3].map((item) => (
              <div key={item} className="card h-24" />
            ))}
          </div>
          <div className="card h-40 animate-pulse" />
        </div>
      )}

      {error && (
        <div className="card border-red-500/50 py-10 mb-8">
          <p className="text-red-300 font-medium mb-2">Benchmark report could not load</p>
          <p className="text-sm text-neural-400">
            {error instanceof Error
              ? error.message
              : 'The API did not return a benchmark report.'}
          </p>
        </div>
      )}

      {runMutation.error && (
        <div className="card border-red-500/50 py-6 mb-8">
          <p className="text-red-300 font-medium mb-2">Benchmark run failed</p>
          <p className="text-sm text-neural-400">
            {runMutation.error instanceof Error
              ? runMutation.error.message
              : 'The benchmark runner returned an unexpected error.'}
          </p>
        </div>
      )}

      {/* Results summary */}
      {report && !isLoading && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="card text-center">
              <div className="text-3xl font-bold text-accent-cyan">
                {(avgPrecision * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-neural-400">Avg Precision@5</div>
            </div>
            <div className="card text-center">
              <div className="text-3xl font-bold text-accent-violet">
                {(avgRecall * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-neural-400">Avg Label Recall@10</div>
            </div>
            <div className="card text-center">
              <div className={`text-3xl font-bold ${passRate >= 70 ? 'text-accent-emerald' : 'text-amber-400'}`}>
                {passRate.toFixed(0)}%
              </div>
              <div className="text-sm text-neural-400">Pass Rate</div>
            </div>
            <div className="card text-center">
              <div className="text-3xl font-bold text-neural-300">
                {report.queries_with_results ?? report.passed_queries}/{report.total_queries}
              </div>
              <div className="text-sm text-neural-400">Queries With Results</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <MetricStrip label="Task match" value={report.avg_task_match_rate} />
            <MetricStrip label="Modality match" value={report.avg_modality_match_rate} />
            <MetricStrip label="Behavior match" value={report.avg_behavior_match_rate} />
          </div>

          {/* Progress bar */}
          <div className="card mb-8">
            <h3 className="text-sm font-medium text-neural-300 mb-3">Overall Health</h3>
            <div className="flex items-center gap-4">
              <div className="flex-1 h-4 bg-neural-800 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    passRate >= 70 ? 'bg-accent-emerald' : passRate >= 50 ? 'bg-accent-amber' : 'bg-red-500'
                  }`}
                  style={{ width: `${passRate}%` }}
                />
              </div>
              <span className="text-neural-300 font-medium w-16 text-right">
                {passRate.toFixed(0)}%
              </span>
            </div>
          </div>

          {((report.summary_warnings && report.summary_warnings.length > 0) ||
            (report.recommendations && report.recommendations.length > 0)) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
              {report.summary_warnings && report.summary_warnings.length > 0 && (
                <section className="card border-amber-500/30">
                  <h3 className="font-medium text-amber-300 mb-3">Benchmark Warnings</h3>
                  <ul className="space-y-2 text-sm text-neural-400">
                    {report.summary_warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </section>
              )}
              {report.recommendations && report.recommendations.length > 0 && (
                <section className="card border-accent-cyan/30">
                  <h3 className="font-medium text-accent-cyan mb-3">Recommendations</h3>
                  <ul className="space-y-2 text-sm text-neural-400">
                    {report.recommendations.map((recommendation) => (
                      <li key={recommendation}>{recommendation}</li>
                    ))}
                  </ul>
                </section>
              )}
            </div>
          )}

          {/* Query results */}
          <h2 className="text-lg font-semibold text-neural-100 mb-4">
            Query Evaluations ({report.query_evaluations?.length || 0})
          </h2>

          <div className="space-y-4">
            {report.query_evaluations?.map((evaluation) => (
              <div
                key={evaluation.query_id}
                className={`card ${
                  evaluation.passed
                    ? 'border-emerald-500/30'
                    : 'border-red-500/30'
                }`}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      {evaluation.passed ? (
                        <CheckCircleIcon className="w-5 h-5 text-emerald-400" />
                      ) : (
                        <XCircleIcon className="w-5 h-5 text-red-400" />
                      )}
                      <h3 className="font-medium text-neural-100">{evaluation.query}</h3>
                    </div>

                    {/* Expected vs Found */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-neural-500">Expected tasks: </span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {evaluation.expected_tasks.map((t) => (
                            <span
                              key={t}
                              className={`badge ${
                                evaluation.found_tasks.includes(t)
                                  ? 'bg-emerald-500/20 text-emerald-400'
                                  : 'bg-red-500/20 text-red-400'
                              }`}
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div>
                        <span className="text-neural-500">Expected modalities: </span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {evaluation.expected_modalities.map((m) => (
                            <span
                              key={m}
                              className={`badge ${
                                evaluation.found_modalities.includes(m)
                                  ? 'bg-emerald-500/20 text-emerald-400'
                                  : 'bg-red-500/20 text-red-400'
                              }`}
                            >
                              {m}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Scores */}
                  <div className="ml-4 flex gap-4">
                    <div className="text-center">
                      <div className="text-xl font-bold text-accent-cyan">
                        {(evaluation.precision_at_5 * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-neural-500">P@5</div>
                    </div>
                    <div className="text-center">
                      <div className="text-xl font-bold text-accent-violet">
                        {(evaluation.label_recall * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-neural-500">Recall</div>
                    </div>
                  </div>
                </div>

                {/* Top results */}
                {evaluation.top_results && evaluation.top_results.length > 0 && (
                  <div className="border-t border-neural-800 pt-3 mt-3">
                    <h4 className="text-xs text-neural-500 mb-2">Top Results:</h4>
                    <div className="flex flex-wrap gap-2">
                      {evaluation.top_results.slice(0, 5).map((r, i) => (
                        <Link
                          key={r.dataset_id}
                          to={`/datasets/${r.dataset_id}`}
                          className="flex items-center gap-1 text-xs bg-neural-800 rounded px-2 py-1 hover:bg-neural-700 transition-colors"
                        >
                          <span className="text-neural-500">#{i + 1}</span>
                          <span className="text-neural-300 truncate max-w-[150px]">
                            {r.title}
                          </span>
                          <span className="text-accent-cyan">{(r.score * 100).toFixed(0)}</span>
                          <ChevronRightIcon className="w-3 h-3 text-neural-600" />
                        </Link>
                      ))}
                    </div>
                  </div>
                )}

                {evaluation.warnings && evaluation.warnings.length > 0 && (
                  <div className="border-t border-neural-800 pt-3 mt-3">
                    <h4 className="text-xs text-amber-300 mb-2">Warnings</h4>
                    <ul className="space-y-1 text-xs text-neural-500">
                      {evaluation.warnings.slice(0, 3).map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Empty state */}
          {(!report.query_evaluations || report.query_evaluations.length === 0) && (
            <div className="card text-center py-12">
              <p className="text-neural-400 mb-4">No evaluation results yet</p>
              <p className="text-sm text-neural-500">
                Click "Run Benchmark" to evaluate search quality
              </p>
            </div>
          )}
        </>
      )}

      {/* Info box */}
      <div className="mt-8 card bg-neural-800/50">
        <h3 className="font-medium text-neural-200 mb-2">About Evaluation</h3>
        <p className="text-sm text-neural-400">
          The evaluation system runs predefined benchmark queries against the search
          engine and measures precision@k and recall. Results help identify areas
          where the ontology matching or ranking needs improvement. A query "passes"
          when it achieves both precision@5 {">"} 40% and label recall {">"} 50%.
        </p>
        <p className="text-sm text-neural-500 mt-2">
          Benchmark queries are defined in{' '}
          <code className="text-accent-cyan">benchmark_queries.yaml</code>
        </p>
      </div>
    </div>
  )
}

function MetricStrip({ label, value }: { label: string; value?: number }) {
  const percent = value === undefined ? 0 : value * 100
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-neural-400">{label}</span>
        <span className="text-sm font-medium text-neural-200">
          {value === undefined ? 'n/a' : `${percent.toFixed(0)}%`}
        </span>
      </div>
      <div className="h-2 rounded-full bg-neural-800 overflow-hidden">
        <div
          className="h-full bg-accent-cyan transition-all duration-500"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  )
}
