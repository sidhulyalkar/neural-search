import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { SearchPage } from './pages/SearchPage'
import { ResultsPage } from './pages/ResultsPage'
import { DatasetPage } from './pages/DatasetPage'
import { OntologyPage } from './pages/OntologyPage'
import { EvaluationPage } from './pages/EvaluationPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/search" element={<ResultsPage />} />
        <Route path="/datasets/:id" element={<DatasetPage />} />
        <Route path="/ontology" element={<OntologyPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
      </Routes>
    </Layout>
  )
}

export default App
