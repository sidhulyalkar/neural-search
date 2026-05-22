import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { getOntology } from '../api/search'
import { SpinnerIcon, SearchIcon, ChevronRightIcon } from '../components/Icons'
import type { Task } from '../types'

export function OntologyPage() {
  const [search, setSearch] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [expandedTask, setExpandedTask] = useState<string | null>(null)
  const navigate = useNavigate()

  const { data: ontology, isLoading } = useQuery({
    queryKey: ['ontology'],
    queryFn: getOntology,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <SpinnerIcon className="w-8 h-8 text-accent-cyan" />
      </div>
    )
  }

  const tasks = ontology?.tasks || []

  // Get unique categories
  const categories = [...new Set(tasks.map((t) => t.category))].sort()

  // Filter tasks
  const filteredTasks = tasks.filter((task) => {
    const matchesSearch =
      !search ||
      task.label.toLowerCase().includes(search.toLowerCase()) ||
      task.synonyms.some((s) => s.toLowerCase().includes(search.toLowerCase()))

    const matchesCategory = !selectedCategory || task.category === selectedCategory

    return matchesSearch && matchesCategory
  })

  const handleTaskSearch = (task: Task) => {
    navigate(`/search?q=${encodeURIComponent(task.label)}`)
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-neural-100 mb-2">Behavioral Task Ontology</h1>
        <p className="text-neural-400">
          Browse the taxonomy of behavioral tasks to find relevant datasets
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar filters */}
        <div className="space-y-6">
          {/* Search */}
          <div className="relative">
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search tasks..."
              className="input pl-10"
            />
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neural-400" />
          </div>

          {/* Categories */}
          <div className="card">
            <h3 className="text-sm font-medium text-neural-300 mb-3">Categories</h3>
            <div className="space-y-1">
              <button
                onClick={() => setSelectedCategory(null)}
                className={`w-full text-left px-3 py-2 rounded text-sm ${
                  !selectedCategory
                    ? 'bg-neural-700 text-accent-cyan'
                    : 'text-neural-400 hover:bg-neural-800 hover:text-neural-200'
                }`}
              >
                All categories
              </button>
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`w-full text-left px-3 py-2 rounded text-sm ${
                    selectedCategory === cat
                      ? 'bg-neural-700 text-accent-cyan'
                      : 'text-neural-400 hover:bg-neural-800 hover:text-neural-200'
                  }`}
                >
                  {cat.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Task list */}
        <div className="lg:col-span-3">
          <div className="text-sm text-neural-500 mb-4">
            {filteredTasks.length} tasks
          </div>

          <div className="space-y-3">
            {filteredTasks.map((task) => (
              <div key={task.id} className="card">
                {/* Task header */}
                <div
                  className="flex items-center justify-between cursor-pointer"
                  onClick={() => setExpandedTask(expandedTask === task.id ? null : task.id)}
                >
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-medium text-neural-100">{task.label}</h3>
                      <span className="badge bg-neural-700 text-neural-400">
                        {task.category.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <p className="text-sm text-neural-400 line-clamp-2">{task.definition}</p>
                  </div>
                  <ChevronRightIcon
                    className={`w-5 h-5 text-neural-500 transition-transform ${
                      expandedTask === task.id ? 'rotate-90' : ''
                    }`}
                  />
                </div>

                {/* Expanded content */}
                {expandedTask === task.id && (
                  <div className="mt-4 pt-4 border-t border-neural-700 space-y-4">
                    {/* Synonyms */}
                    {task.synonyms.length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium text-neural-300 mb-2">Synonyms</h4>
                        <div className="flex flex-wrap gap-2">
                          {task.synonyms.map((syn) => (
                            <span key={syn} className="badge bg-neural-800 text-neural-400">
                              {syn}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Common events */}
                    {task.common_events.length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium text-neural-300 mb-2">Common Events</h4>
                        <div className="flex flex-wrap gap-2">
                          {task.common_events.map((event) => (
                            <span key={event} className="badge bg-neural-800 text-accent-cyan/80">
                              {event}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Modalities */}
                    {task.relevant_modalities.length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium text-neural-300 mb-2">Relevant Modalities</h4>
                        <div className="flex flex-wrap gap-2">
                          {task.relevant_modalities.map((mod) => (
                            <span key={mod} className="badge bg-neural-800 text-accent-violet/80">
                              {mod}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Search button */}
                    <button
                      onClick={() => handleTaskSearch(task)}
                      className="btn-primary mt-4"
                    >
                      Search datasets for {task.label}
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
