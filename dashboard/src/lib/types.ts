export type Action = 'LOCAL' | 'CLOUD'

export interface InferenceEvent {
  id: number
  received_at: string
  telemetry_schema_version: string
  device_id: string
  request_id: string
  timestamp_ms: number
  window_id: number
  predicted_class: string
  predicted_class_id: number
  true_label: string | null
  confidence: number
  uncertainty: number
  requested_action: string
  execution_mode: Action
  split_point: number | null
  failover: boolean
  failure_reason: string
  failure_stage: string
  wifi_connected: boolean
  mqtt_connected: boolean
  rssi: number | null
  rtt_ms: number | null
  rtt_source: string
  rtt_age_ms: number
  free_heap_bytes: number
  free_heap_ratio: number | null
  energy_budget: number | null
  local_inference_ms: number
  request_elapsed_ms: number | null
  edge_compute_ms: number | null
  network_ms: number | null
  server_compute_ms: number | null
  total_latency_ms: number | null
  bytes_tx: number
  bytes_rx: number
  model_version: string
  policy_version: string
  firmware_version: string
  success: boolean
  controlled: boolean
  raw_event?: Record<string, unknown>
}

export interface DashboardSummary {
  api_version: string
  window_event_count: number
  latest_event_id: number | null
  latest_received_at: string | null
  action_counts: Record<Action, number>
  gesture_counts: Record<string, number>
  failover_count: number
  failover_reasons: Record<string, number>
  success_count: number
  controlled_count: number
  confidence_mean: number | null
  uncertainty_mean: number | null
  local_inference_ms: { mean: number | null; p95: number | null }
  request_elapsed_ms: { mean: number | null; p95: number | null }
  server_compute_ms: { mean: number | null; p95: number | null }
  rtt_ms: { mean: number | null; p95: number | null }
  free_heap_bytes: { mean: number | null; latest: number | null }
  bytes_tx_total: number
  bytes_rx_total: number
  unmeasured_fields: string[]
  device_id: string | null
  production_only: boolean
}

export interface DeviceInfo {
  device_id: string
  event_count: number
  latest_event_id: number
  last_seen: string | null
}

export interface DashboardHealth {
  status: string
  phase: string
  milestone: string
  checkpoint: string
  server_version: string
  api_version: string
  ui_version: string
  database_configured: boolean
  event_count: number | null
}
