import { Download, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { InferenceEvent } from '../lib/types'

function percent(value: number) { return `${(value * 100).toFixed(1)}%` }
function valueOrDash(value: number | null, unit = '') { return value == null ? '—' : `${value.toFixed(2)}${unit}` }

export function EventTable({ events, onSelect }: { events: InferenceEvent[]; onSelect: (event: InferenceEvent) => void }) {
  const [query, setQuery] = useState('')
  const [action, setAction] = useState<'ALL' | 'LOCAL' | 'CLOUD'>('ALL')
  const filtered = useMemo(() => events.filter((event) => {
    const text = `${event.request_id} ${event.predicted_class} ${event.failure_reason}`.toLowerCase()
    return (action === 'ALL' || event.execution_mode === action) && text.includes(query.toLowerCase())
  }), [events, query, action])

  const exportCsv = () => {
    const columns = ['id','received_at','device_id','request_id','predicted_class','confidence','uncertainty','requested_action','execution_mode','failover','failure_reason','rtt_ms','free_heap_bytes','local_inference_ms','request_elapsed_ms','server_compute_ms','bytes_tx','bytes_rx','model_version','policy_version','firmware_version'] as const
    const body = [columns.join(','), ...filtered.map((event) => columns.map((column) => JSON.stringify(event[column] ?? '')).join(','))].join('\n')
    const blob = new Blob([body], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `tinyml-events-${new Date().toISOString().replace(/[:.]/g, '-')}.csv`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section className="panel event-table-panel">
      <div className="panel-heading table-heading">
        <div><span className="eyebrow">Decision history</span><h2>Production Event Stream</h2></div>
        <div className="table-tools">
          <label className="search-box"><Search size={15}/><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search request, gesture, failure…"/></label>
          <select value={action} onChange={(event) => setAction(event.target.value as typeof action)}><option>ALL</option><option>LOCAL</option><option>CLOUD</option></select>
          <button className="icon-button" onClick={exportCsv} title="Export visible rows to CSV"><Download size={16}/></button>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <thead><tr><th>ID</th><th>Time</th><th>Gesture</th><th>Decision</th><th>Confidence</th><th>Uncertainty</th><th>RTT</th><th>Local inference</th><th>Failover</th></tr></thead>
          <tbody>
            {filtered.slice(0, 160).map((event) => (
              <tr key={event.id} onClick={() => onSelect(event)}>
                <td className="mono">#{event.id}</td>
                <td>{new Date(event.received_at).toLocaleTimeString()}</td>
                <td><strong>{event.predicted_class}</strong></td>
                <td><span className={`action-pill ${event.execution_mode.toLowerCase()}`}>{event.execution_mode}</span></td>
                <td>{percent(event.confidence)}</td>
                <td>{percent(event.uncertainty)}</td>
                <td>{valueOrDash(event.rtt_ms, ' ms')}</td>
                <td>{valueOrDash(event.local_inference_ms, ' ms')}</td>
                <td>{event.failover ? <span className="warning-pill">{event.failure_reason}</span> : <span className="muted">No</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
