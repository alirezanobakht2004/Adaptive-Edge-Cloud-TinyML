import { useCallback, useEffect, useRef, useState } from 'react'
import { api, dashboardWebSocketUrl } from '../lib/api'
import type { DashboardHealth, DashboardSummary, DeviceInfo, InferenceEvent } from '../lib/types'

const MAX_LIVE_EVENTS = 360

function newestEventId(events: InferenceEvent[]): number {
  return events.reduce((latest, event) => Math.max(latest, event.id), 0)
}

export function useDashboard(deviceId: string | null) {
  const [events, setEvents] = useState<InferenceEvent[]>([])
  const [summary, setSummary] = useState<DashboardSummary | null>(null)
  const [health, setHealth] = useState<DashboardHealth | null>(null)
  const [devices, setDevices] = useState<DeviceInfo[]>([])
  const [connected, setConnected] = useState(false)
  const [snapshotReady, setSnapshotReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const retryRef = useRef(0)
  const latestIdRef = useRef(0)

  const refresh = useCallback(async () => {
    try {
      const [healthData, devicesData, eventsData, summaryData] = await Promise.all([
        api.health(),
        api.devices(true),
        api.events(deviceId, true, 240),
        api.summary(deviceId, true, 500),
      ])
      latestIdRef.current = newestEventId(eventsData.events)
      setHealth(healthData)
      setDevices(devicesData.devices)
      setEvents(eventsData.events)
      setSummary(summaryData)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [deviceId])

  // Load the current database snapshot before opening the live socket. This prevents
  // replaying the full telemetry history from after_id=0 on every page load.
  useEffect(() => {
    let cancelled = false
    setSnapshotReady(false)
    setConnected(false)
    latestIdRef.current = 0

    void refresh().finally(() => {
      if (!cancelled) setSnapshotReady(true)
    })

    return () => {
      cancelled = true
    }
  }, [deviceId, refresh])

  // Keep aggregate cards fresh and provide REST fallback if the socket is unavailable.
  useEffect(() => {
    const timer = window.setInterval(() => {
      void api.summary(deviceId, true, 500).then(setSummary).catch(() => undefined)
      void api.health().then(setHealth).catch(() => undefined)
      if (!connected) {
        void api.events(deviceId, true, 240)
          .then((data) => {
            latestIdRef.current = newestEventId(data.events)
            setEvents(data.events)
          })
          .catch(() => undefined)
      }
    }, 5000)
    return () => window.clearInterval(timer)
  }, [connected, deviceId])

  useEffect(() => {
    if (!snapshotReady) return undefined

    let socket: WebSocket | null = null
    let retryTimer: number | undefined
    let cancelled = false

    const connect = () => {
      if (cancelled) return
      socket = new WebSocket(dashboardWebSocketUrl(deviceId, latestIdRef.current))
      socket.onopen = () => {
        retryRef.current = 0
        setConnected(true)
        setError(null)
      }
      socket.onmessage = (message) => {
        const payload = JSON.parse(message.data) as { type: string; event?: InferenceEvent; detail?: string }
        if (payload.type === 'event' && payload.event) {
          latestIdRef.current = Math.max(latestIdRef.current, payload.event.id)
          setEvents((current) => {
            if (current.some((event) => event.id === payload.event!.id)) return current
            return [payload.event!, ...current].slice(0, MAX_LIVE_EVENTS)
          })
        } else if (payload.type === 'error') {
          setError(payload.detail ?? 'WebSocket error')
        }
      }
      socket.onclose = () => {
        setConnected(false)
        if (!cancelled) {
          const delay = Math.min(8000, 700 * 2 ** retryRef.current++)
          retryTimer = window.setTimeout(connect, delay)
        }
      }
      socket.onerror = () => socket?.close()
    }

    connect()
    return () => {
      cancelled = true
      if (retryTimer) window.clearTimeout(retryTimer)
      socket?.close()
    }
  }, [deviceId, snapshotReady])

  return { events, summary, health, devices, connected, error, refresh }
}
