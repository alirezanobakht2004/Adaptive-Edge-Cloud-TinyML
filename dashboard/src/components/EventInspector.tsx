import { X } from 'lucide-react'
import type { InferenceEvent } from '../lib/types'

export function EventInspector({ event, onClose }: { event: InferenceEvent | null; onClose: () => void }) {
  if (!event) return null
  const rows = [
    ['Request ID', event.request_id], ['Window', event.window_id], ['Gesture', event.predicted_class],
    ['Requested action', event.requested_action], ['Execution mode', event.execution_mode],
    ['Failover', event.failover ? event.failure_reason : 'No'], ['Failure stage', event.failure_stage],
    ['Wi-Fi', event.wifi_connected ? 'Connected' : 'Unavailable'], ['MQTT', event.mqtt_connected ? 'Connected' : 'Unavailable'],
    ['RTT', event.rtt_ms == null ? 'Not measured' : `${event.rtt_ms.toFixed(2)} ms`], ['RTT source', event.rtt_source],
    ['Free heap', `${(event.free_heap_bytes / 1024).toFixed(1)} KB`], ['Local inference', `${event.local_inference_ms.toFixed(3)} ms`],
    ['Request elapsed', event.request_elapsed_ms == null ? 'Not applicable / unavailable' : `${event.request_elapsed_ms.toFixed(2)} ms`],
    ['Server compute', event.server_compute_ms == null ? 'Not applicable / unavailable' : `${event.server_compute_ms.toFixed(2)} ms`],
    ['TX / RX', `${event.bytes_tx} / ${event.bytes_rx} bytes`], ['Model', event.model_version], ['Policy', event.policy_version], ['Firmware', event.firmware_version],
  ]
  return (
    <div className="inspector-backdrop" onMouseDown={onClose}>
      <aside className="inspector" onMouseDown={(e) => e.stopPropagation()}>
        <div className="inspector-head"><div><span className="eyebrow">Event inspector</span><h2>Decision #{event.id}</h2></div><button className="icon-button" onClick={onClose}><X size={18}/></button></div>
        <div className="inspector-grid">{rows.map(([label, value]) => <div key={String(label)}><span>{label}</span><strong>{String(value)}</strong></div>)}</div>
        <div className="provenance-box"><strong>Measurement integrity</strong><p>RSSI, energy, pure network latency and total E2E latency are not populated by the current production telemetry contract. The dashboard leaves them unavailable rather than estimating them.</p></div>
      </aside>
    </div>
  )
}
