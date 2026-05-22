import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { SearchIcon } from '../components/Icons'

const exampleQueries = [
  'Go/NoGo task with neuropixels',
  'Two-alternative forced choice in mice',
  'Calcium imaging decision making',
  'Working memory prefrontal cortex',
  'Reward prediction error dopamine',
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
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-12rem)]">
      <div className="w-full max-w-3xl px-4">
        {/* Hero text */}
        <div className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-bold mb-4">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-cyan to-accent-violet">
              Neural Search
            </span>
          </h1>
          <p className="text-xl text-neural-300">
            Describe your experiment. Find reusable datasets, papers, and starter analyses.
          </p>
        </div>

        {/* Search form */}
        <form onSubmit={handleSearch} className="mb-8">
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What neural data are you looking for?"
              className="input text-lg py-4 pl-12 pr-4"
              autoFocus
            />
            <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neural-400" />
            <button
              type="submit"
              className="absolute right-2 top-1/2 -translate-y-1/2 btn-primary"
            >
              Search
            </button>
          </div>
        </form>

        {/* Example queries */}
        <div className="space-y-3">
          <p className="text-sm text-neural-500 text-center">Try searching for:</p>
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
        <div className="mt-16 grid grid-cols-3 gap-8 text-center">
          <div>
            <div className="text-3xl font-bold text-accent-cyan">DANDI</div>
            <div className="text-sm text-neural-400">Dandisets indexed</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-accent-violet">OpenNeuro</div>
            <div className="text-sm text-neural-400">BIDS datasets</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-accent-emerald">OpenAlex</div>
            <div className="text-sm text-neural-400">Papers linked</div>
          </div>
        </div>
      </div>
    </div>
  )
}
