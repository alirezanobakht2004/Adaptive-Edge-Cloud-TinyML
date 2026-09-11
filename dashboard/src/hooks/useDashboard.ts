import { useCallback, useEffect, useRef, useState } from 'react'
import { api, dashboardPoseWebSocketUrl, dashboardWebSocketUrl } from '../lib/api'
import type { DashboardHealth, DashboardSummary, DeviceInfo, DevicePose, InferenceEvent } from '../lib/types'

const MAX_LIVE_EVENTS = 360

function newestEventId(events: InferenceEvent[]): number {
  return events.reduce((latest, event) => Math.max(latest, event.id), 0)
}

export function useDashboard(deviceId: string | null) {
  const [events, setEvents] = useState<InferenceEvent[]>([])
  const [pose, setPose] = useState<DevicePose | null>(null)
  const [summary, setSummary] = useState<DashboardSummary | null>(null)
  const [health, setHealth] = useState<DashboardHealth | null>(null)
  const [devices, setDevices] = useState<DeviceInfo[]>([])
  const [connected, setConnected] = useState(false)
  const [poseConnected, setPoseConnected] = useState(false)
  const [snapshotReady, setSnapshotReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const retryRef = useRef(0)
  const poseRetryRef = useRef(0)
  const latestIdRef = useRef(0)
  const latestPoseIdRef = useRef(0)

  const refresh = useCallback(async () => {
    try {
      const [healthData, devicesData, eventsData, summaryData, poseData] = await Promise.all([
        api.health(),
        api.devices(true),
        api.events(deviceId, true, 240),
        api.summary(deviceId, true, 500),
        api.poseLatest(deviceId),
      ])
      latestIdRef.current = newestEventId(eventsData.events)
      latestPoseIdRef.current = poseData?.id ?? 0
      setHealth(healthData)
      setDevices(devicesData.devices)
      setEvents(eventsData.events)
      setSummary(summaryData)
      setPose(poseData)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [deviceId])

  // Load the current database snapshot before opening either live socket. This
  // prevents replaying the full decision or pose history from after_id=0.
  useEffect(() => {
    let cancelled = false
    setSnapshotReady(false)
    setConnected(false)
    setPoseConnected(false)
    setPose(null)
    latestIdRef.current = 0
    latestPoseIdRef.current = 0

    void refresh().finally(() => {
      if (!cancelled) setSnapshotReady(true)
    })

    return () => {
      cancelled = true
    }
  }, [deviceId, refresh])

  // Keep aggregate cards fresh and provide REST fallback if the decision socket is unavailable.
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

  // Pose uses its own faster fallback because the 3D twin is intended to respond
  // continuously even when the dedicated pose WebSocket reconnects.
  useEffect(() => {
    const timer = window.setInterval(() => {
      if (poseConnected) return
      void api.poseLatest(deviceId)
        .then((value) => {
          if (!value) return
          if (value.id >= latestPoseIdRef.current) {
            latestPoseIdRef.current = value.id
            setPose(value)
          }
        })
        .catch(() => undefined)
    }, 1000)
    return () => window.clearInterval(timer)
  }, [deviceId, poseConnected])

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
          setError(payload.detail ?? 'Decision WebSocket error')
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

  useEffect(() => {
    if (!snapshotReady) return undefined

    let socket: WebSocket | null = null
    let retryTimer: number | undefined
    let cancelled = false

    const connect = () => {
      if (cancelled) return
      socket = new WebSocket(dashboardPoseWebSocketUrl(deviceId, latestPoseIdRef.current))
      socket.onopen = () => {
        poseRetryRef.current = 0
        setPoseConnected(true)
      }
      socket.onmessage = (message) => {
        const payload = JSON.parse(message.data) as { type: string; pose?: DevicePose; detail?: string }
        if (payload.type === 'pose' && payload.pose) {
          latestPoseIdRef.current = Math.max(latestPoseIdRef.current, payload.pose.id)
          setPose(payload.pose)
        } else if (payload.type === 'error') {
          setError(payload.detail ?? 'Pose WebSocket error')
        }
      }
      socket.onclose = () => {
        setPoseConnected(false)
        if (!cancelled) {
          const delay = Math.min(8000, 500 * 2 ** poseRetryRef.current++)
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

  return { events, pose, summary, health, devices, connected, poseConnected, error, refresh }
}
