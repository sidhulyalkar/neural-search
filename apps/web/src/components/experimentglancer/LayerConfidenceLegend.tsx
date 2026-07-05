const LEGEND_ITEMS: Array<{ status: string; label: string; swatchClass: string }> = [
  {
    status: 'available',
    label: 'File-verified — read from the dataset’s own files',
    swatchClass: 'border-accent-emerald/60 bg-accent-emerald/20',
  },
  {
    status: 'probable',
    label: 'Inferred — guessed from metadata, not yet file-validated',
    swatchClass: 'border-accent-cyan/60 bg-accent-cyan/20',
  },
  {
    status: 'placeholder',
    label: 'Requested, not generated — the query asked for it but nothing produces it yet',
    swatchClass: 'border-dashed border-neural-600 bg-transparent',
  },
  {
    status: 'unsupported',
    label: 'Unsupported — no path to this layer for this dataset',
    swatchClass: 'border-neural-800 bg-neural-900',
  },
]

/** A persistent legend so "probable" is never mistaken for "available" --
 * the single most important trust signal in the whole scene contract. */
export function LayerConfidenceLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[11px] text-neural-500">
      {LEGEND_ITEMS.map((item) => (
        <div key={item.status} className="flex items-center gap-1.5">
          <span className={`inline-block h-2.5 w-2.5 rounded-sm border ${item.swatchClass}`} />
          <span>
            <span className="text-neural-400">{item.status}</span>
            {' — '}
            {item.label}
          </span>
        </div>
      ))}
    </div>
  )
}
