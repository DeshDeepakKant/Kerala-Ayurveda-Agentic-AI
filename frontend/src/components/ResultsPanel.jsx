import { useState } from 'react'

const STATUS_CONFIG = {
  CORRECT:             { icon:'✅', label:'Correct',    cls:'val-green' },
  INCORRECT:           { icon:'❌', label:'Incorrect',  cls:'val-red' },
  INCORRECT_RECOVERED: { icon:'🔄', label:'Recovered',  cls:'val-amber' },
  AMBIGUOUS:           { icon:'⚠️', label:'Ambiguous',  cls:'val-amber' },
  DIRECT_RETRIEVAL:    { icon:'📖', label:'Direct',     cls:'val-blue' },
}

export default function ResultsPanel({ result, error, loading }) {
  if (loading) {
    return (
      <div className="empty-state">
        <span className="spinner" style={{ width:36, height:36, borderWidth:3 }} />
        <h3>Thinking…</h3>
        <p>Running CRAG retrieval and synthesizing your answer with Gemini.</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="error-card">
        <span>⚠️</span>
        <span>{error}</span>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="empty-state">
        <span className="empty-icon">🌿</span>
        <h3>Ask anything about Ayurveda</h3>
        <p>Choose an example from the sidebar, or type your own question about herbs, doshas, or products.</p>
      </div>
    )
  }

  const { crag_status, confidence, sources = [], answer } = result
  const statusInfo = STATUS_CONFIG[crag_status] ?? { icon:'ℹ️', label: crag_status, cls:'val-blue' }
  const confClass = confidence >= 0.7 ? 'val-green' : confidence >= 0.35 ? 'val-amber' : 'val-red'

  return (
    <div className="results-area">
      {/* Metric Pills */}
      <div className="metric-row">
        <div className="metric-pill">
          <span className="metric-icon">{statusInfo.icon}</span>
          <div className="metric-info">
            <span className="mkey">CRAG Status</span>
            <span className={`mval ${statusInfo.cls}`}>{statusInfo.label}</span>
          </div>
        </div>

        <div className="metric-pill">
          <span className="metric-icon">🎯</span>
          <div className="metric-info">
            <span className="mkey">Confidence</span>
            <span className={`mval ${confClass}`}>{(confidence * 100).toFixed(0)}%</span>
          </div>
        </div>

        <div className="metric-pill">
          <span className="metric-icon">📚</span>
          <div className="metric-info">
            <span className="mkey">Sources</span>
            <span className="mval val-blue">{sources.length}</span>
          </div>
        </div>
      </div>

      {/* Synthesized Answer */}
      {answer && (
        <div className="answer-wrap">
          <div className="answer-header">
            <span>✦</span> Synthesized Answer
          </div>
          <div className="answer-divider" />
          <div className="answer-body">
            {answer.split('\n\n').filter(Boolean).map((para, i) => (
              <p key={i} dangerouslySetInnerHTML={{ __html: formatParagraph(para) }} />
            ))}
          </div>
        </div>
      )}

      {/* Source cards */}
      {sources.length > 0 && (
        <div>
          <div className="sources-header">
            <span>📄 Retrieved Sources</span>
            <span className="sources-count">{sources.length}</span>
          </div>
          <div className="source-list">
            {sources.map((src, i) => (
              <SourceCard key={i} source={src} index={i} defaultOpen={i === 0} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function SourceCard({ source, index, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen)

  const rawSrc = source.source || ''
  // Derive a readable title from the source path
  const fileName = rawSrc.split('/').pop().replace(/\.[^.]+$/, '').replace(/_/g, ' ')
  const title = source.title || (fileName && fileName !== 'Unknown' ? fileName : null) || `Source ${index + 1}`
  const score  = source.score
  const text   = source.text || ''
  const docType = source.doc_type || ''

  return (
    <div className="source-item">
      <div className="source-head" onClick={() => setOpen(o => !o)}>
        <span className="source-num">{index + 1}</span>
        <span className="source-title" title={title}>{title}</span>
        {score !== undefined && <span className="source-score">{score.toFixed(4)}</span>}
        <span className={`source-chevron${open ? ' open' : ''}`}>▼</span>
      </div>

      {open && (
        <div className="source-body">
          {docType && (
            <div className="source-tags">
              <span className="source-tag">{docType}</span>
            </div>
          )}
          {text && <SourceText text={text} />}
        </div>
      )}
    </div>
  )
}

/** Bold **text** and simple newline → <br> */
function formatParagraph(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
}

/** Color-coded source text renderer */
function SourceText({ text }) {
  const lines = text.split('\n')
  return (
    <div className="source-text-rich">
      {lines.map((line, i) => {
        // === SECTION HEADER ===
        if (/^={3}\s*.+\s*={3}$/.test(line.trim())) {
          const label = line.trim().replace(/^={3}\s*/, '').replace(/\s*={3}$/, '')
          const colorMap = {
            'DOCUMENT CONTEXT': 'src-hdr-blue',
            'CONTENT':          'src-hdr-green',
            'METADATA':         'src-hdr-amber',
          }
          const cls = colorMap[label.toUpperCase()] || 'src-hdr-purple'
          return (
            <div key={i} className={`src-hdr ${cls}`}>
              <span className="src-hdr-bar"/>
              <span className="src-hdr-text">{label}</span>
              <span className="src-hdr-bar"/>
            </div>
          )
        }
        // Key: Value field lines
        const fieldMatch = line.match(/^([A-Z][\w ]+):\s*(.*)$/)
        if (fieldMatch) {
          return (
            <div key={i} className="src-field">
              <span className="src-field-key">{fieldMatch[1]}:</span>
              <span className="src-field-val">{fieldMatch[2]}</span>
            </div>
          )
        }
        // Bullet points
        if (/^\s*[-•]\s+/.test(line)) {
          return (
            <div key={i} className="src-bullet">
              <span className="src-bullet-dot">▸</span>
              <span>{line.replace(/^\s*[-•]\s+/, '')}</span>
            </div>
          )
        }
        // Blockquote / positioning lines
        if (/^\s*>/.test(line)) {
          return (
            <div key={i} className="src-quote">
              {line.replace(/^\s*>\s*/, '')}
            </div>
          )
        }
        // Separator dashes
        if (/^-{2,}$/.test(line.trim())) {
          return <hr key={i} className="src-sep" />
        }
        // Empty lines
        if (!line.trim()) return <div key={i} className="src-gap" />
        // Regular content
        return <div key={i} className="src-line">{line}</div>
      })}
    </div>
  )
}
