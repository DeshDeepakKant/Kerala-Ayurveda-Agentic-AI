export default function Header({ systemReady }) {
  return (
    <header className="header">
      <div className="header-logo">🌿</div>
      <div className="header-text">
        <h1>Kerala Ayurveda Q&amp;A</h1>
        <p>Powered by CRAG · Hybrid Retrieval · Knowledge Graphs · Gemini 2.5 Flash</p>
      </div>
      <div className="header-badge">
        <span className="header-badge-dot" />
        {systemReady ? 'System Ready' : 'Initializing…'}
      </div>
    </header>
  )
}
