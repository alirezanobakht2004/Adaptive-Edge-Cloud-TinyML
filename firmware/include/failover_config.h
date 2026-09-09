#pragma once
// Mirrors config/r1_failover_v1.json; checked by the desktop contract test.
namespace failover_config {
constexpr const char* VERSION = "r1-failover-v1";
constexpr unsigned CLOUD_RESPONSE_TIMEOUT_MS = 3000;
constexpr unsigned RTT_PROBE_TIMEOUT_MS = 3000;
constexpr unsigned RTT_PROBE_INTERVAL_MS = 5000;
constexpr unsigned RECONNECT_INTERVAL_MS = 5000;
constexpr unsigned MQTT_SOCKET_TIMEOUT_SECONDS = 1;
constexpr unsigned PENDING_CAPACITY = 16;
constexpr unsigned WORK_QUEUE_CAPACITY = 16;
}
