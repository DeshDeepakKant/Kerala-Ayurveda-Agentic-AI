import { useState, useEffect } from 'react'
import Header from './components/Header.jsx'
import Sidebar from './components/Sidebar.jsx'
import QueryPanel from './components/QueryPanel.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'

const API_BASE = '/api'

const EXAMPLE_QUESTIONS = [
  'What are the benefits of Ashwagandha?',
  'How does Triphala help with digestion?',
  'Can pregnant women use Ashwagandha?',
  'Compare Ashwagandha and Brahmi for stress',
  'What is Vata dosha?',
]

export default function App() {
  const [query, setQuery] = useState('')
  const [useCrag, setUseCrag] = useState(true)
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  const [systemReady, setSystemReady] = useState(false)

  // Fetch system health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`)
        if (res.ok) {
          const data = await res.json()
          setStats(data)
          setSystemReady(true)
        }
      } catch {
        // Backend not ready yet, retry
        setTimeout(checkHealth, 3000)
      }
    }
    checkHealth()
  }, [])

  const handleSearch = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          use_crag: useCrag,
          top_k: topK
        })
      })

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setQuery('')
    setResult(null)
    setError(null)
  }

  const handleExampleClick = (q) => {
    setQuery(q)
  }

  return (
    <div className="app-layout">
      <Header systemReady={systemReady} />

      <Sidebar
        useCrag={useCrag}
        setUseCrag={setUseCrag}
        topK={topK}
        setTopK={setTopK}
        stats={stats}
        examples={EXAMPLE_QUESTIONS}
        onExampleClick={handleExampleClick}
      />

      <main className="main-content">
        <QueryPanel
          query={query}
          setQuery={setQuery}
          loading={loading}
          onSearch={handleSearch}
          onClear={handleClear}
        />

        <ResultsPanel
          result={result}
          error={error}
          loading={loading}
        />
      </main>
    </div>
  )
}
