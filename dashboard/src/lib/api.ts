import type { DashboardHealth, DashboardSummary, DeviceInfo, DevicePose, InferenceEvent } from './types'

async function json<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: 'application/json' } })
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
  return response.json() as Promise<T>
}

async function optionalJson<T>(path: string): Promise<T | null> {
  const response = await fetch(path, { headers: { Accept: 'application/json' } })
  if (response.status === 404) return null
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
  return response.json() as Promise<T>
}

export const api = {
  health: () => json<DashboardHealth>('/api/dashboard/health'),
  devices: (productionOnly = true) =>
    json<{ devices: DeviceInfo[] }>(`/api/dashboard/devices?production_only=${productionOnly}`),
  events: (deviceId: string | null, productionOnly = true, limit = 240) => {
    const params = new URLSearchParams({ limit: String(limit), production_only: String(productionOnly) })
    if (deviceId) params.set('device_id', deviceId)
    return json<{ events: InferenceEvent[]; count: number; next_before_id: number | null }>(
      `/api/dashboard/events?${params}`,
    )
  },
  latest: (deviceId: string | null, productionOnly = true) => {
    const params = new URLSearchParams({ production_only: String(productionOnly) })
    if (deviceId) params.set('device_id', deviceId)
    return json<InferenceEvent>(`/api/dashboard/latest?${params}`)
  },
  poseLatest: (deviceId: string | null) => {
    const params = new URLSearchParams()
    if (deviceId) params.set('device_id', deviceId)
    return optionalJson<DevicePose>(`/api/dashboard/pose/latest?${params}`)
  },
  summary: (deviceId: string | null, productionOnly = true, windowEvents = 500) => {
    const params = new URLSearchParams({
      production_only: String(productionOnly),
      window_events: String(windowEvents),
    })
    if (deviceId) params.set('device_id', deviceId)
    return json<DashboardSummary>(`/api/dashboard/summary?${params}`)
  },
}

export function dashboardWebSocketUrl(deviceId: string | null, afterId: number): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const params = new URLSearchParams({ production_only: 'true', after_id: String(afterId) })
  if (deviceId) params.set('device_id', deviceId)
  return `${protocol}//${window.location.host}/ws/dashboard/events?${params}`
}


export function dashboardPoseWebSocketUrl(deviceId: string | null, afterId: number): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const params = new URLSearchParams({ after_id: String(afterId) })
  if (deviceId) params.set('device_id', deviceId)
  return `${protocol}//${window.location.host}/ws/dashboard/pose?${params}`
}
