import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { DashboardSummary, InferenceEvent } from '../lib/types'

const tooltipStyle = {
  background: '#101722',
  border: '1px solid rgba(255,255,255,.1)',
  borderRadius: 12,
  color: '#eef4ff',
}

const GESTURE_CLASSES = ['IDLE', 'SWIPE_LEFT', 'SWIPE_RIGHT', 'ROTATE_CW', 'SHAKE'] as const

function shortTime(value: string) {
  const date = new Date(value)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function ConfidenceChart({ events }: { events: InferenceEvent[] }) {
  const data = [...events].reverse().slice(-120).map((event) => ({
    id: event.id,
    time: shortTime(event.received_at),
    confidence: event.confidence * 100,
    uncertainty: event.uncertainty * 100,
  }))
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}>
        <defs>
          <linearGradient id="confidenceFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#45e0a1" stopOpacity={0.32} />
            <stop offset="95%" stopColor="#45e0a1" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="uncertaintyFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#7f97ff" stopOpacity={0.28} />
            <stop offset="95%" stopColor="#7f97ff" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(255,255,255,.05)" vertical={false} />
        <XAxis dataKey="time" minTickGap={40} tick={{ fill: '#697a91', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 100]} tick={{ fill: '#697a91', fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle} formatter={(value: number | string) => `${Number(value).toFixed(1)}%`} />
        <Legend wrapperStyle={{ fontSize: 12, color: '#8c9bb0' }} />
        <Area type="monotone" dataKey="confidence" name="Confidence" stroke="#45e0a1" fill="url(#confidenceFill)" strokeWidth={2} />
        <Area type="monotone" dataKey="uncertainty" name="Uncertainty" stroke="#7f97ff" fill="url(#uncertaintyFill)" strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export function ActionDonut({ summary }: { summary: DashboardSummary | null }) {
  const data = [
    { name: 'LOCAL', value: summary?.action_counts.LOCAL ?? 0, color: '#45e0a1' },
    { name: 'CLOUD', value: summary?.action_counts.CLOUD ?? 0, color: '#7f97ff' },
  ]
  const total = data.reduce((sum, item) => sum + item.value, 0)
  return (
    <div className="donut-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} dataKey="value" innerRadius="67%" outerRadius="88%" paddingAngle={total > 0 ? 3 : 0} stroke="none">
            {data.map((item) => <Cell key={item.name} fill={item.color} />)}
          </Pie>
          <Tooltip contentStyle={tooltipStyle} />
        </PieChart>
      </ResponsiveContainer>
      <div className="donut-center"><strong>{total > 0 ? total : '—'}</strong><span>{total > 0 ? 'decisions' : 'no events'}</span></div>
    </div>
  )
}

export function LatencyChart({ events }: { events: InferenceEvent[] }) {
  const data = [...events].reverse().slice(-100).map((event) => ({
    time: shortTime(event.received_at),
    local: event.local_inference_ms,
    request: event.request_elapsed_ms,
    server: event.server_compute_ms,
  }))
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 12, right: 10, left: -18, bottom: 0 }}>
        <CartesianGrid stroke="rgba(255,255,255,.05)" vertical={false} />
        <XAxis dataKey="time" minTickGap={45} tick={{ fill: '#697a91', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: '#697a91', fontSize: 11 }} axisLine={false} tickLine={false} unit=" ms" />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 12, color: '#8c9bb0' }} />
        <Line type="monotone" dataKey="local" name="Local inference" stroke="#45e0a1" dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="request" name="Cloud request elapsed" stroke="#7f97ff" dot={false} strokeWidth={2} connectNulls={false} />
        <Line type="monotone" dataKey="server" name="Server compute" stroke="#d799ff" dot={false} strokeWidth={2} connectNulls={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function RttHeapChart({ events }: { events: InferenceEvent[] }) {
  const data = [...events].reverse().slice(-100).map((event) => ({
    time: shortTime(event.received_at),
    rtt: event.rtt_ms,
    heap: event.free_heap_bytes / 1024,
  }))
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 12, right: 20, left: -15, bottom: 0 }}>
        <CartesianGrid stroke="rgba(255,255,255,.05)" vertical={false} />
        <XAxis dataKey="time" minTickGap={45} tick={{ fill: '#697a91', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis yAxisId="left" tick={{ fill: '#697a91', fontSize: 11 }} axisLine={false} tickLine={false} unit=" ms" />
        <YAxis yAxisId="right" orientation="right" tick={{ fill: '#697a91', fontSize: 11 }} axisLine={false} tickLine={false} unit=" KB" />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 12, color: '#8c9bb0' }} />
        <Line yAxisId="left" type="monotone" dataKey="rtt" name="RTT probe" stroke="#ffbd69" dot={false} strokeWidth={2} connectNulls={false} />
        <Line yAxisId="right" type="monotone" dataKey="heap" name="Free heap" stroke="#55c9ff" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function GestureChart({ summary }: { summary: DashboardSummary | null }) {
  const data = GESTURE_CLASSES.map((gesture) => ({
    gesture,
    value: summary?.gesture_counts[gesture] ?? 0,
  }))
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 2 }}>
        <CartesianGrid stroke="rgba(255,255,255,.05)" vertical={false} />
        <XAxis dataKey="gesture" interval={0} tick={{ fill: '#697a91', fontSize: 9 }} axisLine={false} tickLine={false} />
        <YAxis allowDecimals={false} tick={{ fill: '#697a91', fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle} />
        <Bar dataKey="value" name="Observed windows" radius={[7, 7, 2, 2]} fill="#55c9ff" />
      </BarChart>
    </ResponsiveContainer>
  )
}
