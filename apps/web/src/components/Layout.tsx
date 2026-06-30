import { Link, useLocation } from 'react-router-dom'
import { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
}

const navLinks = [
  { path: '/', label: 'Search', exact: true },
  { path: '/atlas', label: 'Atlas' },
  { path: '/methods', label: 'Methods' },
  { path: '/labs/neatlabs', label: 'NEATLabs' },
  { path: '/demo', label: 'Demo' },
  { path: '/graph', label: 'Knowledge Graph' },
  { path: '/ontology', label: 'Ontology' },
  { path: '/coverage', label: 'Coverage' },
  { path: '/reports', label: 'Reports' },
  { path: '/evaluation', label: 'Evaluation' },
]

export function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="min-h-screen flex flex-col bg-neural-950 text-neural-100">
      <header className="border-b border-neural-800/50 sticky top-0 z-50 bg-neural-950/95 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between py-3">
            <Link
              to="/"
              className="font-mono text-sm font-medium text-neural-200 tracking-tight hover:text-white transition-colors"
            >
              neural search
            </Link>

            <nav className="flex items-center">
              {navLinks.map((link, i) => {
                const isActive = link.exact
                  ? location.pathname === link.path
                  : location.pathname === link.path || location.pathname.startsWith(link.path + '/')
                return (
                  <span key={link.path} className="flex items-center">
                    {i > 0 && (
                      <span className="text-neural-700 mx-1 text-xs select-none">/</span>
                    )}
                    <Link
                      to={link.path}
                      className={`px-2 py-1 text-sm transition-colors ${
                        isActive
                          ? 'text-accent-cyan'
                          : 'text-neural-500 hover:text-neural-200'
                      }`}
                    >
                      {link.label}
                    </Link>
                  </span>
                )
              })}
            </nav>
          </div>
        </div>
      </header>

      <main className="flex-1">
        {children}
      </main>

      <footer className="border-t border-neural-800/40 py-8 mt-24">
        <div className="max-w-5xl mx-auto px-6 lg:px-8 flex items-center justify-between">
          <span className="font-mono text-xs text-neural-600">neural search</span>
          <span className="text-xs text-neural-700">
            experiment-aware neuroscience data discovery
          </span>
        </div>
      </footer>
    </div>
  )
}
