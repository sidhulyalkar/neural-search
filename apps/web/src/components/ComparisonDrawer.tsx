import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { XIcon, DownloadIcon, SpinnerIcon, TableIcon, CompareIcon } from './Icons'
import { compareDatasets, exportComparisonMarkdown } from '../api/search'
import type { ComparisonResult, DatasetComparisonItem, FieldComparison } from '../types'

interface ComparisonDrawerProps {
  selectedIds: string[]
  onRemove: (id: string) => void
  onClear: () => void
  isOpen: boolean
  onClose: () => void
}

export function ComparisonDrawer({
  selectedIds,
  onRemove,
  onClear,
  isOpen,
  onClose,
}: ComparisonDrawerProps) {
  const [activeTab, setActiveTab] = useState<'summary' | 'table' | 'details'>('summary')

  const {
    data: comparison,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['comparison', selectedIds],
    queryFn: () => compareDatasets(selectedIds),
    enabled: selectedIds.length >= 2 && isOpen,
  })

  const exportMutation = useMutation({
    mutationFn: () => exportComparisonMarkdown(selectedIds),
    onSuccess: (blob) => {
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `comparison_${selectedIds.slice(0, 3).join('_')}.md`
      a.click()
      URL.revokeObjectURL(url)
    },
  })

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-neural-950/80 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="absolute inset-y-0 right-0 w-full max-w-4xl bg-neural-900 border-l border-neural-800 shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neural-800">
          <div className="flex items-center gap-3">
            <CompareIcon className="w-5 h-5 text-accent-cyan" />
            <h2 className="text-lg font-semibold text-neural-100">
              Dataset Comparison
            </h2>
            <span className="badge badge-cyan">{selectedIds.length} datasets</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => exportMutation.mutate()}
              disabled={exportMutation.isPending || selectedIds.length < 2}
              className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
            >
              <DownloadIcon className="w-4 h-4" />
              {exportMutation.isPending ? 'Exporting...' : 'Export Markdown'}
            </button>
            <button
              onClick={onClear}
              className="btn-secondary text-xs py-1.5 px-3"
            >
              Clear All
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded hover:bg-neural-800 transition-colors"
            >
              <XIcon className="w-5 h-5 text-neural-400" />
            </button>
          </div>
        </div>

        {/* Selected datasets pills */}
        <div className="px-6 py-3 border-b border-neural-800 flex flex-wrap gap-2">
          {selectedIds.map((id) => (
            <span
              key={id}
              className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-neural-800 text-sm text-neural-200"
            >
              {comparison?.datasets.find((d) => d.dataset_id === id)?.title?.slice(0, 30) || id}
              <button
                onClick={() => onRemove(id)}
                className="p-0.5 rounded hover:bg-neural-700 transition-colors"
              >
                <XIcon className="w-3 h-3 text-neural-400" />
              </button>
            </span>
          ))}
        </div>

        {/* Tabs */}
        <div className="px-6 py-2 border-b border-neural-800 flex gap-4">
          {(['summary', 'table', 'details'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-2 text-sm font-medium rounded transition-colors ${
                activeTab === tab
                  ? 'bg-accent-cyan/10 text-accent-cyan'
                  : 'text-neural-400 hover:text-neural-100'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {selectedIds.length < 2 ? (
            <div className="text-center py-16 text-neural-400">
              <CompareIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg mb-2">Select at least 2 datasets to compare</p>
              <p className="text-sm text-neural-500">
                Use the checkboxes on search results to add datasets
              </p>
            </div>
          ) : isLoading ? (
            <div className="flex items-center justify-center py-20">
              <SpinnerIcon className="w-8 h-8 text-accent-cyan" />
              <span className="ml-3 text-neural-400">Loading comparison...</span>
            </div>
          ) : error ? (
            <div className="text-center py-16 text-red-400">
              <p>Error loading comparison</p>
              <p className="text-sm text-neural-500 mt-2">
                {error instanceof Error ? error.message : 'Unknown error'}
              </p>
            </div>
          ) : comparison ? (
            <>
              {activeTab === 'summary' && <SummaryView comparison={comparison} />}
              {activeTab === 'table' && <TableView comparison={comparison} />}
              {activeTab === 'details' && <DetailsView comparison={comparison} />}
            </>
          ) : null}
        </div>
      </div>
    </div>
  )
}

function SummaryView({ comparison }: { comparison: ComparisonResult }) {
  const { summary } = comparison

  return (
    <div className="space-y-6">
      {/* Key insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Shared characteristics */}
        {summary.shared_tasks.length > 0 && (
          <div className="rounded border border-accent-cyan/30 bg-accent-cyan/5 p-4">
            <h3 className="text-sm font-medium text-accent-cyan mb-2">Shared Tasks</h3>
            <div className="flex flex-wrap gap-1.5">
              {summary.shared_tasks.map((task) => (
                <span key={task} className="badge bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30">
                  {task.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        )}

        {summary.shared_modalities.length > 0 && (
          <div className="rounded border border-accent-violet/30 bg-accent-violet/5 p-4">
            <h3 className="text-sm font-medium text-accent-violet mb-2">Shared Modalities</h3>
            <div className="flex flex-wrap gap-1.5">
              {summary.shared_modalities.map((mod) => (
                <span key={mod} className="badge bg-accent-violet/10 text-accent-violet border border-accent-violet/30">
                  {mod.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Readiness ranking */}
      <div className="rounded border border-neural-800 bg-neural-950/60 p-4">
        <h3 className="text-sm font-medium text-neural-200 mb-3">Analysis Readiness Ranking</h3>
        <div className="space-y-2">
          {summary.readiness_ranking.map((item, index) => (
            <div
              key={item.dataset_id}
              className="flex items-center gap-3 p-2 rounded bg-neural-900/50"
            >
              <span className="w-6 h-6 flex items-center justify-center rounded-full bg-accent-emerald/10 text-accent-emerald text-sm font-medium">
                {index + 1}
              </span>
              <span className="flex-1 text-sm text-neural-200 truncate">{item.title}</span>
              <span className="text-lg font-semibold text-accent-emerald">{item.score}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Common vs different */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {summary.common_fields.length > 0 && (
          <div className="rounded border border-accent-emerald/30 bg-accent-emerald/5 p-4">
            <h3 className="text-sm font-medium text-accent-emerald mb-2">Common Ground</h3>
            <ul className="text-sm text-neural-300 space-y-1">
              {summary.common_fields.map((field) => (
                <li key={field}>• {field}</li>
              ))}
            </ul>
          </div>
        )}

        {summary.different_fields.length > 0 && (
          <div className="rounded border border-amber-500/30 bg-amber-500/5 p-4">
            <h3 className="text-sm font-medium text-amber-400 mb-2">Key Differences</h3>
            <ul className="text-sm text-neural-300 space-y-1">
              {summary.different_fields.map((field) => (
                <li key={field}>• {field}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* All unique values */}
      <div className="rounded border border-neural-800 bg-neural-950/60 p-4">
        <h3 className="text-sm font-medium text-neural-200 mb-3">Combined Coverage</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-neural-500">All tasks:</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {summary.all_tasks.map((t) => (
                <span key={t} className="badge bg-neural-800 text-neural-300">
                  {t.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
          <div>
            <span className="text-neural-500">All modalities:</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {summary.all_modalities.map((m) => (
                <span key={m} className="badge bg-neural-800 text-neural-300">
                  {m.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function TableView({ comparison }: { comparison: ComparisonResult }) {
  const { datasets, field_comparisons } = comparison

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-neural-800">
            <th className="text-left py-3 px-4 text-neural-400 font-medium sticky left-0 bg-neural-900 min-w-40">
              Field
            </th>
            {datasets.map((ds) => (
              <th
                key={ds.dataset_id}
                className="text-left py-3 px-4 text-neural-200 font-medium min-w-48"
              >
                {ds.title.slice(0, 40)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {field_comparisons.map((fc) => (
            <FieldComparisonRow
              key={fc.field_name}
              fieldComparison={fc}
              datasets={datasets}
            />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function FieldComparisonRow({
  fieldComparison,
  datasets,
}: {
  fieldComparison: FieldComparison
  datasets: DatasetComparisonItem[]
}) {
  const { field_label, values, all_same } = fieldComparison

  return (
    <tr className="border-b border-neural-800/50 hover:bg-neural-800/20">
      <td className="py-3 px-4 text-neural-300 font-medium sticky left-0 bg-neural-900">
        {field_label}
        {all_same && (
          <span className="ml-2 text-xs text-accent-emerald">same</span>
        )}
      </td>
      {datasets.map((ds) => {
        const value = values[ds.dataset_id]
        return (
          <td key={ds.dataset_id} className="py-3 px-4 text-neural-200">
            {formatCellValue(value)}
          </td>
        )
      })}
    </tr>
  )
}

function formatCellValue(value: unknown): React.ReactNode {
  if (value === null || value === undefined) {
    return <span className="text-neural-600">-</span>
  }
  if (typeof value === 'boolean') {
    return value ? (
      <span className="text-accent-emerald">Yes</span>
    ) : (
      <span className="text-neural-500">No</span>
    )
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-neural-600">-</span>
    }
    return (
      <div className="flex flex-wrap gap-1">
        {value.slice(0, 3).map((v, i) => (
          <span key={i} className="badge bg-neural-800 text-neural-300 text-xs">
            {String(v).replace(/_/g, ' ')}
          </span>
        ))}
        {value.length > 3 && (
          <span className="text-xs text-neural-500">+{value.length - 3}</span>
        )}
      </div>
    )
  }
  return String(value)
}

function DetailsView({ comparison }: { comparison: ComparisonResult }) {
  const { datasets } = comparison

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {datasets.map((ds) => (
        <DatasetDetailCard key={ds.dataset_id} dataset={ds} />
      ))}
    </div>
  )
}

function DatasetDetailCard({ dataset }: { dataset: DatasetComparisonItem }) {
  return (
    <div className="rounded border border-neural-800 bg-neural-950/60 p-4">
      <h3 className="text-lg font-medium text-neural-100 mb-1">{dataset.title}</h3>
      <div className="text-xs text-neural-500 mb-4">
        {dataset.source.toUpperCase()} · {dataset.source_id}
      </div>

      <div className="space-y-4 text-sm">
        {/* Readiness */}
        <div className="flex items-center justify-between">
          <span className="text-neural-400">Analysis Readiness</span>
          <span className="text-lg font-semibold text-accent-emerald">
            {dataset.analysis_readiness_score}/100
          </span>
        </div>

        {/* Labels */}
        {dataset.task_labels.length > 0 && (
          <div>
            <span className="text-neural-500 text-xs">Tasks</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {dataset.task_labels.map((t) => (
                <span key={t} className="badge bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30">
                  {t.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        )}

        {dataset.modalities.length > 0 && (
          <div>
            <span className="text-neural-500 text-xs">Modalities</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {dataset.modalities.map((m) => (
                <span key={m} className="badge bg-accent-violet/10 text-accent-violet border border-accent-violet/30">
                  {m.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Strengths */}
        {dataset.strengths.length > 0 && (
          <div>
            <span className="text-neural-500 text-xs">Strengths</span>
            <ul className="mt-1 text-neural-300 text-xs space-y-0.5">
              {dataset.strengths.slice(0, 3).map((s, i) => (
                <li key={i}>• {s}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Limitations */}
        {dataset.limitations.length > 0 && (
          <div>
            <span className="text-neural-500 text-xs">Limitations</span>
            <ul className="mt-1 text-neural-300 text-xs space-y-0.5">
              {dataset.limitations.slice(0, 3).map((l, i) => (
                <li key={i}>• {l}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Missing metadata */}
        {dataset.missing_metadata.length > 0 && (
          <div>
            <span className="text-neural-500 text-xs">Missing Metadata</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {dataset.missing_metadata.map((m) => (
                <span key={m} className="badge bg-amber-500/10 text-amber-400 border border-amber-500/30 text-xs">
                  {m}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Suggested analyses */}
        {dataset.suggested_analyses.length > 0 && (
          <div>
            <span className="text-neural-500 text-xs">Suggested Analyses</span>
            <ul className="mt-1 text-neural-300 text-xs space-y-0.5">
              {dataset.suggested_analyses.slice(0, 3).map((a, i) => (
                <li key={i}>• {a}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

// Floating comparison bar component for use in results page
export function ComparisonBar({
  selectedIds,
  selectedTitles,
  onRemove,
  onClear,
  onCompare,
}: {
  selectedIds: string[]
  selectedTitles: Record<string, string>
  onRemove: (id: string) => void
  onClear: () => void
  onCompare: () => void
}) {
  if (selectedIds.length === 0) return null

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 bg-neural-900 border border-neural-700 rounded-lg shadow-2xl px-4 py-3 flex items-center gap-4 max-w-3xl">
      <div className="flex items-center gap-2">
        <CompareIcon className="w-5 h-5 text-accent-cyan" />
        <span className="text-sm text-neural-200 font-medium">
          {selectedIds.length} selected
        </span>
      </div>

      <div className="flex-1 flex flex-wrap gap-1.5 max-w-lg overflow-hidden">
        {selectedIds.slice(0, 3).map((id) => (
          <span
            key={id}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-neural-800 text-xs text-neural-300"
          >
            {(selectedTitles[id] || id).slice(0, 20)}
            <button
              onClick={() => onRemove(id)}
              className="p-0.5 rounded hover:bg-neural-700"
            >
              <XIcon className="w-2.5 h-2.5" />
            </button>
          </span>
        ))}
        {selectedIds.length > 3 && (
          <span className="text-xs text-neural-500">+{selectedIds.length - 3} more</span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onClear}
          className="text-xs text-neural-400 hover:text-neural-100 transition-colors"
        >
          Clear
        </button>
        <button
          onClick={onCompare}
          disabled={selectedIds.length < 2}
          className={`btn-primary text-xs py-1.5 px-4 flex items-center gap-1.5 ${
            selectedIds.length < 2 ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          <TableIcon className="w-4 h-4" />
          Compare
        </button>
      </div>
    </div>
  )
}
