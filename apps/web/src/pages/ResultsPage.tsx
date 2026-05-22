import { useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { SearchIcon, SpinnerIcon, ChevronRightIcon } from '../components/Icons'
import { searchDatasets } from '../api/search'
import { DatasetCard } from '../components/DatasetCard'

export function ResultsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const query = searchParams.get('q') || ''
  const [inputValue, setInputValue] = useState(query)

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', query],
    queryFn: () => searchDatasets(query),
    enabled: !!query,
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (inputValue.trim()) {
      setSearchParams({ q: inputValue.trim() })
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Search bar */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="relative max-w-3xl">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Search for neural datasets..."
            className="input py-3 pl-12 pr-4"
          />
          <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neural-400" />
          <button type="submit" className="absolute right-2 top-1/2 -translate-y-1/2 btn-primary">
            Search
          </button>
        </div>
      </form>

      {/* Results */}
      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <SpinnerIcon className="w-8 h-8 text-accent-cyan" />
          <span className="ml-3 text-neural-400">Searching...</span>
        </div>
      )}

      {error && (
        <div className="card border-red-500/50 text-center py-10">
          <p className="text-red-400">Error loading results. Please try again.</p>
        </div>
      )}

      {data && (
        <>
          {/* Results header */}
          <div className="flex items-center justify-between mb-6">
            <p className="text-neural-400">
              Found <span className="text-neural-100 font-medium">{data.total_count}</span> datasets
              {data.search_time_ms && (
                <span className="text-neural-500"> in {data.search_time_ms.toFixed(0)}ms</span>
              )}
            </p>

            {/* Filters could go here */}
          </div>

          {/* Results list */}
          {data.results.length > 0 ? (
            <div className="space-y-4">
              {data.results.map((result) => (
                <DatasetCard key={result.dataset.id} result={result} />
              ))}
            </div>
          ) : (
            <div className="card text-center py-16">
              <p className="text-neural-400 mb-4">No datasets found for "{query}"</p>
              <p className="text-sm text-neural-500">
                Try using different keywords or browse the{' '}
                <Link to="/ontology" className="text-accent-cyan hover:underline">
                  ontology
                </Link>
              </p>
            </div>
          )}

          {/* Facets sidebar could go here */}
        </>
      )}
    </div>
  )
}
