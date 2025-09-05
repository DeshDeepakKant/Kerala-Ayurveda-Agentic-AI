export default function Sidebar({
  useCrag, setUseCrag,
  topK, setTopK,
  stats,
  examples, onExampleClick
}) {
  return (
    <aside className="sidebar">

      {/* Settings */}
      <div className="sidebar-section">
        <h3>⚙ Settings</h3>

        <div className="toggle-row">
          <label>CRAG Self-healing</label>
          <label className="toggle">
            <input type="checkbox" checked={useCrag} onChange={e => setUseCrag(e.target.checked)} />
            <div className="toggle-track" />
            <div className="toggle-thumb" />
          </label>
        </div>

        <div className="slider-wrap">
          <div className="slider-head">
            <span>Results count</span>
            <span>{topK}</span>
          </div>
          <input type="range" min={1} max={10} value={topK} onChange={e => setTopK(Number(e.target.value))} />
        </div>
      </div>

      {/* Stats */}
      <div className="sidebar-section">
        <h3>📊 System</h3>
        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-label">Documents</span>
            <span className="stat-value">{stats?.indexed_documents ?? '—'}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Chunks</span>
            <span className="stat-value">{stats?.indexed_chunks ?? '—'}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">KG Nodes</span>
            <span className="stat-value">{stats?.knowledge_graph_nodes ?? '—'}</span>
          </div>
        </div>
      </div>

      {/* Examples */}
      <div className="sidebar-section">
        <h3>💡 Try asking</h3>
        <div className="chips">
          {examples.map((q, i) => (
            <button key={i} className="chip" onClick={() => onExampleClick(q)}>{q}</button>
          ))}
        </div>
      </div>

    </aside>
  )
}
