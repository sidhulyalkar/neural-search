import { useState } from 'react'
import { SpinnerIcon, CheckCircleIcon } from '../components/Icons'

interface BenchmarkQuery {
  id: string
  query: string
  expected_tasks: string[]
  expected_modalities: string[]
  notes?: string
}

// These would come from the API in production
const mockBenchmarks: BenchmarkQuery[] = [
  {
    id: '1',
    query: 'Go/NoGo task with neuropixels recording',
    expected_tasks: ['go_nogo'],
    expected_modalities: ['neuropixels', 'extracellular_ephys'],
    notes: 'Should match response inhibition datasets',
  },
  {
    id: '2',
    query: 'Two-alternative forced choice decision making',
    expected_tasks: ['two_alternative_forced_choice'],
    expected_modalities: ['extracellular_ephys', 'calcium_imaging'],
  },
  {
    id: '3',
    query: 'Spatial navigation hippocampus mice',
    expected_tasks: ['spatial_navigation', 'place_cell_recording'],
    expected_modalities: ['extracellular_ephys'],
  },
]

export function EvaluationPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<Record<string, { precision: number; recall: number }>>({})

  const runBenchmark = async () => {
    setIsRunning(true)
    setResults({})

    // Simulate running benchmarks
    for (const benchmark of mockBenchmarks) {
      await new Promise((r) => setTimeout(r, 500))
      setResults((prev) => ({
        ...prev,
        [benchmark.id]: {
          precision: Math.random() * 0.4 + 0.6,
          recall: Math.random() * 0.3 + 0.7,
        },
      }))
    }

    setIsRunning(false)
  }

  const avgPrecision =
    Object.values(results).length > 0
      ? Object.values(results).reduce((a, b) => a + b.precision, 0) / Object.values(results).length
      : 0

  const avgRecall =
    Object.values(results).length > 0
      ? Object.values(results).reduce((a, b) => a + b.recall, 0) / Object.values(results).length
      : 0

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-neural-100 mb-2">Evaluation Dashboard</h1>
        <p className="text-neural-400">
          Run benchmark queries to evaluate search quality
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mb-8">
        <button
          onClick={runBenchmark}
          disabled={isRunning}
          className="btn-primary flex items-center gap-2"
        >
          {isRunning ? (
            <>
              <SpinnerIcon className="w-4 h-4" />
              Running...
            </>
          ) : (
            'Run Benchmark'
          )}
        </button>

        {Object.keys(results).length > 0 && (
          <div className="flex gap-8">
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-cyan">
                {(avgPrecision * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-neural-500">Avg Precision</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-violet">
                {(avgRecall * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-neural-500">Avg Recall</div>
            </div>
          </div>
        )}
      </div>

      {/* Benchmark queries */}
      <div className="space-y-4">
        {mockBenchmarks.map((benchmark) => {
          const result = results[benchmark.id]

          return (
            <div key={benchmark.id} className="card">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-medium text-neural-100 mb-2">
                    {benchmark.query}
                  </h3>

                  <div className="flex flex-wrap gap-4 text-sm">
                    <div>
                      <span className="text-neural-500">Expected tasks: </span>
                      {benchmark.expected_tasks.map((t) => (
                        <span key={t} className="badge badge-cyan ml-1">
                          {t}
                        </span>
                      ))}
                    </div>
                    <div>
                      <span className="text-neural-500">Expected modalities: </span>
                      {benchmark.expected_modalities.map((m) => (
                        <span key={m} className="badge badge-violet ml-1">
                          {m}
                        </span>
                      ))}
                    </div>
                  </div>

                  {benchmark.notes && (
                    <p className="text-sm text-neural-500 mt-2">{benchmark.notes}</p>
                  )}
                </div>

                {/* Results */}
                <div className="ml-4 text-right">
                  {isRunning && !result && (
                    <SpinnerIcon className="w-5 h-5 text-neural-500" />
                  )}
                  {result && (
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <CheckCircleIcon className="w-4 h-4 text-emerald-400" />
                        <span className="text-sm">
                          P: {(result.precision * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <CheckCircleIcon className="w-4 h-4 text-emerald-400" />
                        <span className="text-sm">
                          R: {(result.recall * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Info */}
      <div className="mt-8 card bg-neural-800/50">
        <h3 className="font-medium text-neural-200 mb-2">About Evaluation</h3>
        <p className="text-sm text-neural-400">
          The evaluation system runs predefined benchmark queries against the search
          engine and measures precision@k and recall. Results help identify areas
          where the ontology matching or ranking needs improvement.
        </p>
        <p className="text-sm text-neural-500 mt-2">
          Benchmark queries are defined in <code className="text-accent-cyan">data/eval/benchmark_queries.yaml</code>
        </p>
      </div>
    </div>
  )
}
