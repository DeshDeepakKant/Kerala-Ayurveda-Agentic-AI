const STATUS_MAP = {
  CORRECT:             { label: 'Correct',            cls: 'green' },
  INCORRECT:           { label: 'Incorrect',          cls: 'red' },
  INCORRECT_RECOVERED: { label: 'Recovered',          cls: 'amber' },
  AMBIGUOUS:           { label: 'Ambiguous',          cls: 'amber' },
}

export default function StatusBadge({ status }) {
  const info = STATUS_MAP[status] ?? { label: status, cls: 'amber' }
  return (
    <span className={`metric-value ${info.cls}`}>
      {info.label}
    </span>
  )
}
