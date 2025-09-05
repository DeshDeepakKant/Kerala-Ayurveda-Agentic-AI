export default function QueryPanel({ query, setQuery, loading, onSearch, onClear }) {
  const handleKeyDown = e => { if (e.key === 'Enter' && !e.shiftKey && !loading) onSearch() }

  return (
    <div>
      <p className="query-label">Ask a Question</p>
      <div className="input-wrap">
        <span className="input-icon">🔍</span>
        <input
          className="query-input"
          type="text"
          placeholder="e.g. What are the benefits of Ashwagandha for stress relief?"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          autoFocus
        />
      </div>
      <div className="btn-row">
        <button className="btn btn-primary" onClick={onSearch} disabled={loading || !query.trim()}>
          {loading ? <><span className="spinner" /> Retrieving…</> : <>🔎 Search</>}
        </button>
        <button className="btn btn-ghost" onClick={onClear} disabled={loading}>
          Clear
        </button>
      </div>
    </div>
  )
}
