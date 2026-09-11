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

const UI_VERSION = 'dashboard-ui-r1-v4'

function pct(value: number | null | undefined) { return value == null ? '—' : `${(value * 100).toFixed(1)}%` }
function ms(value: number | null | undefined, digits = 2) { return value == null ? '—' : `${value.toFixed(digits)} ms` }
function kb(value: number | null | undefined) { return value == null ? '—' : `${(value / 1024).toFixed(1)} KB` }
function bytes(value: number | null | undefined) { return value == null ? '—' : new Intl.NumberFormat().format(value) }
function rate(value: number | null) { return value == null ? '—' : `${value.toFixed(1)}%` }
function ageMs(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return '—'
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(1)} s`
}

function freshness(receivedAt?: string | null) {
  if (!receivedAt) return { label: 'NO DATA', className: 'offline', age: '—', fresh: false }
  const parsed = new Date(receivedAt).getTime()
  if (!Number.isFinite(parsed)) return { label: 'NO DATA', className: 'offline', age: '—', fresh: false }
  const seconds = Math.max(0, (Date.now() - parsed) / 1000)
  if (seconds <= 5) return { label: 'LIVE', className: 'live', age: `${seconds.toFixed(1)}s`, fresh: true }
  if (seconds <= 30) return { label: 'STALE', className: 'stale', age: `${seconds.toFixed(0)}s`, fresh: false }
  return { label: 'OFFLINE?', className: 'offline', age: `${Math.floor(seconds / 60)}m`, fresh: false }
}

function MetricCard({ icon, label, value, note, tone = 'default' }: { icon: ReactNode; label: string; value: string; note?: string; tone?: string }) {
  return <div className={`metric-card ${tone}`}><div className="metric-icon">{icon}</div><div><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div></div>
}

export default function App() {
  const [selectedDevice, setSelectedDevice] = useState<string | null>('esp32-r1')
  const [selectedEvent, setSelectedEvent] = useState<InferenceEvent | null>(null)
  const { events, pose, summary, health, devices, connected, poseConnected, error, refresh } = useDashboard(selectedDevice)
  const latest = events[0] ?? null
  const decisionFreshness = freshness(latest?.received_at)
  const poseFreshness = freshness(pose?.received_at)
  const decisionIsCurrent = decisionFreshness.fresh
  const poseIsCurrent = pose != null && poseFreshness.fresh && (Date.now() - new Date(pose.received_at).getTime()) <= 2000
  const hasSummaryWindow = (summary?.window_event_count ?? 0) > 0
  const localRate = hasSummaryWindow && summary ? (summary.action_counts.LOCAL / summary.window_event_count) * 100 : null
  const cloudRate = hasSummaryWindow && summary ? (summary.action_counts.CLOUD / summary.window_event_count) * 100 : null
  const failoverRate = hasSummaryWindow && summary ? (summary.failover_count / summary.window_event_count) * 100 : null
  const successRate = hasSummaryWindow && summary ? (summary.success_count / summary.window_event_count) * 100 : null
  const hasCloudTiming = events.some((event) => event.execution_mode === 'CLOUD' && (event.request_elapsed_ms != null || event.server_compute_ms != null))

  useEffect(() => {
    if (selectedDevice === 'esp32-r1' && devices.length && !devices.some((device) => device.device_id === selectedDevice)) {
      setSelectedDevice(devices[0].device_id)
    }
  }, [devices, selectedDevice])

  const latestVersions = useMemo(() => ({
    firmware: latest?.firmware_version ?? '—', model: latest?.model_version ?? '—', policy: latest?.policy_version ?? '—',
  }), [latest])

  const latestAction = latest?.execution_mode ?? null
  const actionPillClass = !latest ? 'offline' : !decisionIsCurrent ? 'stale' : latestAction?.toLowerCase() ?? 'offline'
  const actionPillText = !latest
    ? 'NO DATA'
    : !decisionIsCurrent
      ? 'DECISION STALE'
      : latest.failover
        ? 'FAILOVER → LOCAL'
        : latest.execution_mode

  const lastConnectivity = latest
    ? `${latest.wifi_connected ? 'Wi-Fi' : 'No Wi-Fi'} / ${latest.mqtt_connected ? 'MQTT' : 'No MQTT'}`
    : 'NO TELEMETRY'
  const connectivityValue = !latest ? 'NO TELEMETRY' : decisionIsCurrent ? lastConnectivity : `LAST: ${lastConnectivity}`
  const connectivityNote = decisionIsCurrent ? 'Live decision telemetry' : `${decisionFreshness.label} · ${decisionFreshness.age}`
  const rttNote = latest?.rtt_ms == null
    ? 'Unavailable'
    : `${latest.rtt_source || 'unknown source'} · age ${ageMs(latest.rtt_age_ms)}`

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><BrainCircuit size={22}/></div><div><strong>EDGE–CLOUD TINYML</strong><span>Adaptive Inference Command Center</span></div></div>
        <div className="topbar-actions">
          <div className={`stream-status ${connected ? 'connected' : ''}`}><span className="pulse-dot"/>{connected ? 'Decision WS live' : 'Decision WS reconnecting'}</div>
          <select value={selectedDevice ?? ''} onChange={(event) => setSelectedDevice(event.target.value || null)}>{devices.map((device) => <option value={device.device_id} key={device.device_id}>{device.device_id}</option>)}</select>
          <button className="icon-button" onClick={() => void refresh()} title="Refresh dashboard"><RefreshCw size={17}/></button>
        </div>
      </header>

      <section className="hero-strip">
        <div><span className="eyebrow">Phase 11 · M11 · Live observability</span><h1>Production Decision Telemetry</h1><p>Binary LOCAL/CLOUD policy telemetry from ESP32-S3 → MQTT → PostgreSQL, with measured fields preserved exactly as recorded.</p></div>
        <div className={`freshness-badge ${decisionFreshness.className}`}><span>{decisionFreshness.label}</span><strong>{decisionFreshness.age}</strong><small>decision telemetry age</small></div>
      </section>

      {error && <div className="error-banner"><TriangleAlert size={17}/><span>{error}</span></div>}

      <section className="metric-grid">
        <MetricCard icon={<Activity/>} label={decisionIsCurrent ? 'Detected gesture' : 'Last detected gesture'} value={latest?.predicted_class ?? '—'} note={`Window ${latest?.window_id ?? '—'} · ${decisionFreshness.label}`} tone="accent"/>
        <MetricCard icon={latest?.execution_mode === 'CLOUD' ? <Cloud/> : <Cpu/>} label={decisionIsCurrent ? 'Execution' : 'Last execution'} value={latest?.execution_mode ?? '—'} note={latest?.failover ? `Failover: ${latest.failure_reason}` : decisionIsCurrent ? 'Learned binary policy' : `Last reported · ${decisionFreshness.age}`} tone={latest?.failover ? 'warning' : 'default'}/>
        <MetricCard icon={<Gauge/>} label="Confidence" value={pct(latest?.confidence)} note={`Mean ${pct(summary?.confidence_mean)}`}/>
        <MetricCard icon={<Zap/>} label="Uncertainty" value={pct(latest?.uncertainty)} note="Normalized predictive entropy"/>
        <MetricCard icon={<Radio/>} label="RTT probe" value={ms(latest?.rtt_ms)} note={rttNote}/>
        <MetricCard icon={<Cpu/>} label="Local inference" value={ms(latest?.local_inference_ms, 3)} note={`P95 ${ms(summary?.local_inference_ms.p95, 3)}`}/>
        <MetricCard icon={<HardDrive/>} label="Free heap" value={kb(latest?.free_heap_bytes)} note="Measured on ESP32"/>
        <MetricCard icon={decisionIsCurrent && latest?.mqtt_connected ? <Wifi/> : <WifiOff/>} label="Connectivity" value={connectivityValue} note={connectivityNote}/>
      </section>

      <section className="main-grid">
        <article className="panel twin-panel">
          <div className="panel-heading"><div><span className="eyebrow">Device twin</span><h2>ESP32-S3 + MPU6050</h2></div><span className={`action-pill ${actionPillClass}`}>{actionPillText}</span></div>
          <DeviceTwin latest={latest} pose={pose} poseConnected={poseConnected}/>
          <div className="twin-state-row">
            <div><span>Wi-Fi</span><strong>{!latest ? 'NO DATA' : decisionIsCurrent ? (latest.wifi_connected ? 'CONNECTED' : 'UNAVAILABLE') : `STALE · LAST ${latest.wifi_connected ? 'CONNECTED' : 'UNAVAILABLE'}`}</strong></div>
            <div><span>MQTT</span><strong>{!latest ? 'NO DATA' : decisionIsCurrent ? (latest.mqtt_connected ? 'CONNECTED' : 'UNAVAILABLE') : `STALE · LAST ${latest.mqtt_connected ? 'CONNECTED' : 'UNAVAILABLE'}`}</strong></div>
            <div><span>Policy</span><strong>{latestVersions.policy}</strong></div>
            <div><span>Device attitude</span><strong>{!pose ? 'NO POSE TELEMETRY' : poseIsCurrent ? `ESTIMATED · ${pose.yaw_reference.toUpperCase()}` : 'POSE STALE · LAST ESTIMATE'}</strong></div>
          </div>
        </article>

        <article className="panel decision-panel">
          <div className="panel-heading"><div><span className="eyebrow">Current decision</span><h2>Inference State</h2></div><ShieldCheck size={20}/></div>
          <div className="decision-summary">
            <div><span>Latest result</span><strong>{latest?.predicted_class ?? 'NO DATA'}</strong><small>{decisionFreshness.label} · {decisionFreshness.age}</small></div>
            <div><span>Effective action</span><strong className={latest?.execution_mode === 'CLOUD' ? 'cloud-text' : 'local-text'}>{latest?.execution_mode ?? '—'}</strong><small>{latest?.failover ? `Failover: ${latest.failure_reason}` : 'Learned binary policy'}</small></div>
          </div>
          <div className="decision-metrics">
            <div><span>Confidence</span><strong>{pct(latest?.confidence)}</strong></div>
            <div><span>Uncertainty</span><strong>{pct(latest?.uncertainty)}</strong></div>
          </div>
          <div className="state-list">
            <div><span>Requested action</span><strong>{latest?.requested_action ?? '—'}</strong></div>
            <div><span>Failover</span><strong>{latest?.failover ? latest.failure_reason : latest ? 'NONE' : '—'}</strong></div>
            <div><span>Request elapsed</span><strong>{ms(latest?.request_elapsed_ms)}</strong></div>
            <div><span>Server compute</span><strong>{ms(latest?.server_compute_ms)}</strong></div>
            <div><span>Transfer TX / RX</span><strong>{bytes(latest?.bytes_tx)} / {bytes(latest?.bytes_rx)} B</strong></div>
            <div><span>RTT provenance</span><strong>{latest?.rtt_ms == null ? 'UNAVAILABLE' : `${latest.rtt_source} · age ${ageMs(latest.rtt_age_ms)}`}</strong></div>
          </div>
        </article>
      </section>

      <section className="charts-grid">
        <article className="panel chart-panel wide"><div className="panel-heading"><div><span className="eyebrow">Model certainty</span><h2>Confidence vs. Uncertainty</h2></div><span className="chart-legend-note">Last 120 events</span></div><div className="chart-body"><ConfidenceChart events={events}/></div></article>
        <article className="panel chart-panel"><div className="panel-heading"><div><span className="eyebrow">Action space</span><h2>LOCAL / CLOUD Mix</h2></div></div><div className="chart-body donut-body"><ActionDonut summary={summary}/></div><div className="mini-stats"><span><i className="dot local"/>LOCAL <strong>{rate(localRate)}</strong></span><span><i className="dot cloud"/>CLOUD <strong>{rate(cloudRate)}</strong></span></div></article>
        <article className="panel chart-panel wide"><div className="panel-heading"><div><span className="eyebrow">Measured timing channels</span><h2>Latency Components</h2></div><span className="chart-legend-note">{hasCloudTiming ? 'Measured channels only · no fabricated E2E/network latency' : 'No CLOUD timing samples in current event window'}</span></div><div className="chart-body"><LatencyChart events={events}/></div></article>
        <article className="panel chart-panel"><div className="panel-heading"><div><span className="eyebrow">Device resources</span><h2>RTT & Free Heap</h2></div></div><div className="chart-body"><RttHeapChart events={events}/></div></article>
        <article className="panel chart-panel"><div className="panel-heading"><div><span className="eyebrow">Observed output classes</span><h2>Gesture Distribution</h2></div><span className="chart-legend-note">Canonical 5-class contract</span></div><div className="chart-body"><GestureChart summary={summary}/></div></article>
        <article className="panel integrity-panel">
          <div className="panel-heading"><div><span className="eyebrow">System integrity</span><h2>Production Contract</h2></div><Database size={20}/></div>
          <div className="integrity-grid"><div><span>Success rate</span><strong>{rate(successRate)}</strong></div><div><span>Failover rate</span><strong>{rate(failoverRate)}</strong></div><div><span>Events in window</span><strong>{summary?.window_event_count ?? '—'}</strong></div><div><span>Decision rows</span><strong>{health?.event_count ?? '—'}</strong></div></div>
          <div className="integrity-tags"><span>Binary LOCAL/CLOUD</span><span>Production only</span><span>No split payloads</span><span>No raw IMU windows</span><span>No energy claim</span><span>Pose rows: {health?.pose_event_count ?? '—'}</span></div>
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
