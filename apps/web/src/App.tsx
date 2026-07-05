import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { SearchPage } from './pages/SearchPage'
import { ResultsPage } from './pages/ResultsPage'
import { DatasetPage } from './pages/DatasetPage'
import { OntologyPage } from './pages/OntologyPage'
import { ReportsPage } from './pages/ReportsPage'
import { EvaluationPage } from './pages/EvaluationPage'
import { DemoPage } from './pages/DemoPage'
import { CoveragePage } from './pages/CoveragePage'
import { BrainAtlasPage } from './pages/BrainAtlasPage'
import { LabShowcasePage } from './pages/LabShowcasePage'
import { MethodsKGPage } from './pages/MethodsKGPage'
import { DisorderMapPage } from './pages/DisorderMapPage'
import { KnowledgeExplorerPage } from './pages/KnowledgeExplorerPage'
import { ExperimentGlancerPage } from './pages/ExperimentGlancerPage'

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
        <Route path="/labs/neatlabs" element={<LabShowcasePage />} />
        <Route path="/methods" element={<MethodsKGPage />} />
        <Route path="/disorders" element={<DisorderMapPage />} />
        <Route path="/experimentglancer" element={<ExperimentGlancerPage />} />
      </Routes>
    </Layout>
  )
}

export default App
