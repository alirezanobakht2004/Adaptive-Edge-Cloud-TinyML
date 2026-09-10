import {
  Activity, BrainCircuit, Cloud, Cpu, Database, Gauge, HardDrive, Radio, RefreshCw,
  Router, ShieldCheck, TriangleAlert, Wifi, WifiOff, Zap,
} from 'lucide-react'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { ActionDonut, ConfidenceChart, GestureChart, LatencyChart, RttHeapChart } from './components/Charts'
import { DeviceTwin } from './components/DeviceTwin'
import { EventInspector } from './components/EventInspector'
import { EventTable } from './components/EventTable'
import { useDashboard } from './hooks/useDashboard'
import type { InferenceEvent } from './lib/types'

const UI_VERSION = 'dashboard-ui-r1-v1'

function pct(value: number | null | undefined) { return value == null ? '—' : `${(value * 100).toFixed(1)}%` }
function ms(value: number | null | undefined, digits = 2) { return value == null ? '—' : `${value.toFixed(digits)} ms` }
function kb(value: number | null | undefined) { return value == null ? '—' : `${(value / 1024).toFixed(1)} KB` }
function bytes(value: number | null | undefined) { return value == null ? '—' : new Intl.NumberFormat().format(value) }

function freshness(receivedAt?: string) {
  if (!receivedAt) return { label: 'NO DATA', className: 'offline', age: '—' }
  const seconds = Math.max(0, (Date.now() - new Date(receivedAt).getTime()) / 1000)
  if (seconds <= 5) return { label: 'LIVE', className: 'live', age: `${seconds.toFixed(1)}s` }
  if (seconds <= 30) return { label: 'STALE', className: 'stale', age: `${seconds.toFixed(0)}s` }
  return { label: 'OFFLINE?', className: 'offline', age: `${Math.floor(seconds / 60)}m` }
}

function MetricCard({ icon, label, value, note, tone = 'default' }: { icon: ReactNode; label: string; value: string; note?: string; tone?: string }) {
  return <div className={`metric-card ${tone}`}><div className="metric-icon">{icon}</div><div><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div></div>
}

export default function App() {
  const [selectedDevice, setSelectedDevice] = useState<string | null>('esp32-r1')
  const [selectedEvent, setSelectedEvent] = useState<InferenceEvent | null>(null)
  const { events, summary, health, devices, connected, error, refresh } = useDashboard(selectedDevice)
  const latest = events[0] ?? null
  const live = freshness(latest?.received_at)
  const localRate = summary?.window_event_count ? (summary.action_counts.LOCAL / summary.window_event_count) * 100 : 0
  const cloudRate = summary?.window_event_count ? (summary.action_counts.CLOUD / summary.window_event_count) * 100 : 0
  const failoverRate = summary?.window_event_count ? (summary.failover_count / summary.window_event_count) * 100 : 0
  const successRate = summary?.window_event_count ? (summary.success_count / summary.window_event_count) * 100 : 0

  useEffect(() => {
    if (selectedDevice === 'esp32-r1' && devices.length && !devices.some((device) => device.device_id === selectedDevice)) {
      setSelectedDevice(devices[0].device_id)
    }
  }, [devices, selectedDevice])

  const latestVersions = useMemo(() => ({
    firmware: latest?.firmware_version ?? '—', model: latest?.model_version ?? '—', policy: latest?.policy_version ?? '—',
  }), [latest])

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><BrainCircuit size={22}/></div><div><strong>EDGE–CLOUD TINYML</strong><span>Adaptive Inference Command Center</span></div></div>
        <div className="topbar-actions">
          <div className={`stream-status ${connected ? 'connected' : ''}`}><span className="pulse-dot"/>{connected ? 'WebSocket live' : 'Reconnecting'}</div>
          <select value={selectedDevice ?? ''} onChange={(event) => setSelectedDevice(event.target.value || null)}>{devices.map((device) => <option value={device.device_id} key={device.device_id}>{device.device_id}</option>)}</select>
          <button className="icon-button" onClick={() => void refresh()} title="Refresh dashboard"><RefreshCw size={17}/></button>
        </div>
      </header>

      <section className="hero-strip">
        <div><span className="eyebrow">Phase 11 · M11 · Live observability</span><h1>Production Decision Telemetry</h1><p>Binary LOCAL/CLOUD policy telemetry from ESP32-S3 → MQTT → PostgreSQL, with measured fields preserved exactly as recorded.</p></div>
        <div className={`freshness-badge ${live.className}`}><span>{live.label}</span><strong>{live.age}</strong><small>telemetry age</small></div>
      </section>

      {error && <div className="error-banner"><TriangleAlert size={17}/><span>{error}</span></div>}

      <section className="metric-grid">
        <MetricCard icon={<Activity/>} label="Detected gesture" value={latest?.predicted_class ?? '—'} note={`Window ${latest?.window_id ?? '—'}`} tone="accent"/>
        <MetricCard icon={latest?.execution_mode === 'CLOUD' ? <Cloud/> : <Cpu/>} label="Execution" value={latest?.execution_mode ?? '—'} note={latest?.failover ? `Failover: ${latest.failure_reason}` : 'Learned binary policy'} tone={latest?.failover ? 'warning' : 'default'}/>
        <MetricCard icon={<Gauge/>} label="Confidence" value={pct(latest?.confidence)} note={`Mean ${pct(summary?.confidence_mean)}`}/>
        <MetricCard icon={<Zap/>} label="Uncertainty" value={pct(latest?.uncertainty)} note="Normalized predictive entropy"/>
        <MetricCard icon={<Radio/>} label="RTT probe" value={ms(latest?.rtt_ms)} note={latest?.rtt_source ?? 'Unavailable'}/>
        <MetricCard icon={<Cpu/>} label="Local inference" value={ms(latest?.local_inference_ms, 3)} note={`P95 ${ms(summary?.local_inference_ms.p95, 3)}`}/>
        <MetricCard icon={<HardDrive/>} label="Free heap" value={kb(latest?.free_heap_bytes)} note="Measured on ESP32"/>
        <MetricCard icon={latest?.mqtt_connected ? <Wifi/> : <WifiOff/>} label="Connectivity" value={`${latest?.wifi_connected ? 'Wi-Fi' : 'No Wi-Fi'} / ${latest?.mqtt_connected ? 'MQTT' : 'No MQTT'}`} note={live.label}/>
      </section>

      <section className="main-grid">
        <article className="panel twin-panel">
          <div className="panel-heading"><div><span className="eyebrow">Device twin</span><h2>ESP32-S3 + MPU6050</h2></div><span className={`action-pill ${latest?.execution_mode?.toLowerCase() ?? 'local'}`}>{latest?.failover ? 'FAILOVER → LOCAL' : latest?.execution_mode ?? 'NO DATA'}</span></div>
          <DeviceTwin latest={latest}/>
          <div className="twin-state-row">
            <div><span>Wi-Fi</span><strong>{latest?.wifi_connected ? 'CONNECTED' : 'UNAVAILABLE'}</strong></div>
            <div><span>MQTT</span><strong>{latest?.mqtt_connected ? 'CONNECTED' : 'UNAVAILABLE'}</strong></div>
            <div><span>Policy</span><strong>{latestVersions.policy}</strong></div>
            <div><span>Physical pose</span><strong>NOT MEASURED</strong></div>
          </div>
        </article>

        <article className="panel decision-panel">
          <div className="panel-heading"><div><span className="eyebrow">Current decision</span><h2>Inference State</h2></div><ShieldCheck size={20}/></div>
          <div className="decision-hero"><span>{latest?.predicted_class ?? 'NO DATA'}</span><strong>{pct(latest?.confidence)}</strong><small>confidence</small></div>
          <div className="decision-bars">
            <div><span>Confidence <strong>{pct(latest?.confidence)}</strong></span><div><i style={{ width: `${(latest?.confidence ?? 0) * 100}%` }}/></div></div>
            <div className="uncertainty"><span>Uncertainty <strong>{pct(latest?.uncertainty)}</strong></span><div><i style={{ width: `${(latest?.uncertainty ?? 0) * 100}%` }}/></div></div>
          </div>
          <div className="state-list">
            <div><span>Requested action</span><strong>{latest?.requested_action ?? '—'}</strong></div>
            <div><span>Effective action</span><strong>{latest?.execution_mode ?? '—'}</strong></div>
            <div><span>Failover</span><strong>{latest?.failover ? latest.failure_reason : 'NONE'}</strong></div>
            <div><span>Request elapsed</span><strong>{ms(latest?.request_elapsed_ms)}</strong></div>
            <div><span>Server compute</span><strong>{ms(latest?.server_compute_ms)}</strong></div>
            <div><span>Transfer TX / RX</span><strong>{bytes(latest?.bytes_tx)} / {bytes(latest?.bytes_rx)} B</strong></div>
          </div>
        </article>
      </section>

      <section className="charts-grid">
        <article className="panel chart-panel wide"><div className="panel-heading"><div><span className="eyebrow">Model certainty</span><h2>Confidence vs. Uncertainty</h2></div><span className="chart-legend-note">Last 120 events</span></div><div className="chart-body"><ConfidenceChart events={events}/></div></article>
        <article className="panel chart-panel"><div className="panel-heading"><div><span className="eyebrow">Action space</span><h2>LOCAL / CLOUD Mix</h2></div></div><div className="chart-body donut-body"><ActionDonut summary={summary}/></div><div className="mini-stats"><span><i className="dot local"/>LOCAL <strong>{localRate.toFixed(1)}%</strong></span><span><i className="dot cloud"/>CLOUD <strong>{cloudRate.toFixed(1)}%</strong></span></div></article>
        <article className="panel chart-panel wide"><div className="panel-heading"><div><span className="eyebrow">Measured timing channels</span><h2>Latency Components</h2></div><span className="chart-legend-note">No fabricated E2E/network latency</span></div><div className="chart-body"><LatencyChart events={events}/></div></article>
        <article className="panel chart-panel"><div className="panel-heading"><div><span className="eyebrow">Device resources</span><h2>RTT & Free Heap</h2></div></div><div className="chart-body"><RttHeapChart events={events}/></div></article>
        <article className="panel chart-panel"><div className="panel-heading"><div><span className="eyebrow">Observed output classes</span><h2>Gesture Distribution</h2></div></div><div className="chart-body"><GestureChart summary={summary}/></div></article>
        <article className="panel integrity-panel">
          <div className="panel-heading"><div><span className="eyebrow">System integrity</span><h2>Production Contract</h2></div><Database size={20}/></div>
          <div className="integrity-grid"><div><span>Success rate</span><strong>{successRate.toFixed(1)}%</strong></div><div><span>Failover rate</span><strong>{failoverRate.toFixed(1)}%</strong></div><div><span>Events in window</span><strong>{summary?.window_event_count ?? 0}</strong></div><div><span>DB total</span><strong>{health?.event_count ?? '—'}</strong></div></div>
          <div className="integrity-tags"><span>Binary LOCAL/CLOUD</span><span>Production only</span><span>No split payloads</span><span>No raw IMU</span><span>No energy claim</span></div>
        </article>
      </section>

      <EventTable events={events} onSelect={setSelectedEvent}/>

      <section className="footer-panels">
        <div className="panel version-panel"><Router/><div><span>Firmware</span><strong>{latestVersions.firmware}</strong></div><div><span>Model</span><strong>{latestVersions.model}</strong></div><div><span>Policy</span><strong>{latestVersions.policy}</strong></div></div>
        <div className="provenance-line"><span>{UI_VERSION}</span><span>{health?.api_version ?? 'API unavailable'}</span><span>PostgreSQL-backed</span><span>Dark command UI</span></div>
      </section>

      <EventInspector event={selectedEvent} onClose={() => setSelectedEvent(null)}/>
    </main>
  )
}
