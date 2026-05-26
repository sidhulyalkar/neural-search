import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { SearchIcon } from '../components/Icons'

const exampleQueries = [
  'Find reversal learning datasets with reward omission and trial outcomes',
  'Go/NoGo task with calcium imaging in mPFC and lick events',
  'Visual decision-making with Neuropixels recordings',
  'Find datasets where I can decode choice from neural activity',
  'Human ECoG or iEEG reaching data for BCI classification',
  'Delay discounting with fiber photometry and reward choice',
]

export function SearchPage() {
  const [query, setQuery] = useState('')
  const navigate = useNavigate()

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`)
    }
  }

  const handleExampleClick = (example: string) => {
    navigate(`/search?q=${encodeURIComponent(example)}`)
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-12rem)] py-10">
      <div className="w-full max-w-5xl px-4">
        {/* Hero text */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 rounded border border-accent-emerald/30 bg-accent-emerald/10 px-3 py-1 text-xs font-medium text-accent-emerald mb-5">
            Experiment-aware search, not generic RAG
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold mb-4">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-cyan to-accent-violet">
              Neural Search
            </span>
          </h1>
          <p className="text-xl text-neural-300 max-w-3xl mx-auto">
            Describe task structure, behavior, modality, and analysis intent. Find reusable datasets with provenance, dataset cards, and starter notebooks.
          </p>
        </div>

        {/* Search form */}
        <form onSubmit={handleSearch} className="mb-8">
          <div className="flex flex-col sm:flex-row gap-3 rounded-lg border border-neural-700 bg-neural-900/80 p-3 shadow-xl shadow-neural-950/30">
            <div className="relative flex-1">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What experiment are you trying to reuse?"
                className="input text-base sm:text-lg py-4 pl-12 pr-4"
                autoFocus
              />
              <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neural-400" />
            </div>
            <button
              type="submit"
              className="btn-primary min-h-12 px-6"
            >
              Search
            </button>
          </div>
        </form>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 mb-10 text-sm">
          {[
            ['Ontology', 'tasks, behaviors, regions'],
            ['Metadata', 'species, modality, standards'],
            ['Embeddings', 'semantic intent matching'],
            ['Provenance', 'papers, warnings, QA'],
          ].map(([label, description]) => (
            <div key={label} className="rounded border border-neural-800 bg-neural-900/60 p-3">
              <div className="font-medium text-neural-100">{label}</div>
              <div className="text-neural-500">{description}</div>
            </div>
          ))}
        </div>

        {/* Example queries */}
        <div className="space-y-3">
          <p className="text-sm text-neural-500 text-center">Demo queries:</p>
          <div className="flex flex-wrap justify-center gap-2">
            {exampleQueries.map((example) => (
              <button
                key={example}
                onClick={() => handleExampleClick(example)}
                className="badge bg-neural-800 text-neural-300 hover:bg-neural-700 hover:text-neural-100 transition-colors cursor-pointer px-3 py-1.5"
              >
                {example}
              </button>
            ))}
          </div>
        </div>

        {/* Stats */}
        <div className="mt-14 grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-accent-cyan">Dataset Cards</div>
            <div className="text-sm text-neural-400">reuse summaries and readiness</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-accent-violet">Starter Notebooks</div>
            <div className="text-sm text-neural-400">first-analysis scaffolds</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-accent-emerald">Benchmarks</div>
            <div className="text-sm text-neural-400">label recovery checks</div>
          </div>
        </div>
      </div>
    </div>
  )
}
