import { Link, useLocation } from 'react-router-dom'
import { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
}

const navLinks = [
  { path: '/', label: 'Search' },
  { path: '/ontology', label: 'Ontology' },
  { path: '/evaluation', label: 'Evaluation' },
]

export function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-neural-800 bg-neural-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-cyan to-accent-violet flex items-center justify-center">
                <span className="text-neural-950 font-bold text-sm">NS</span>
              </div>
              <span className="font-semibold text-lg">Neural Search</span>
            </Link>

            {/* Navigation */}
            <nav className="flex items-center space-x-1">
              {navLinks.map((link) => {
                const isActive = location.pathname === link.path
                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-neural-800 text-accent-cyan'
                        : 'text-neural-300 hover:text-neural-100 hover:bg-neural-800/50'
                    }`}
                  >
                    {link.label}
                  </Link>
                )
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-neural-800 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-neural-500">
            Neural Search MVP - Experiment-aware neural data discovery
          </p>
        </div>
      </footer>
    </div>
  )
}
