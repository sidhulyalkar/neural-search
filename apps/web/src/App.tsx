import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { SearchPage } from './pages/SearchPage'
import { ResultsPage } from './pages/ResultsPage'
import { DatasetPage } from './pages/DatasetPage'
import { OntologyPage } from './pages/OntologyPage'
import { ReportsPage } from './pages/ReportsPage'
import { EvaluationPage } from './pages/EvaluationPage'
import { DemoPage } from './pages/DemoPage'
import { KnowledgeExplorerPage } from './pages/KnowledgeExplorerPage'
import { CoveragePage } from './pages/CoveragePage'
import { BrainAtlasPage } from './pages/BrainAtlasPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/search" element={<ResultsPage />} />
        <Route path="/datasets/:id" element={<DatasetPage />} />
        <Route path="/ontology" element={<OntologyPage />} />
        <Route path="/reports" element={<ReportsPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/graph" element={<KnowledgeExplorerPage />} />
        <Route path="/coverage" element={<CoveragePage />} />
        <Route path="/atlas" element={<BrainAtlasPage />} />
      </Routes>
    </Layout>
  )
}

export default App
