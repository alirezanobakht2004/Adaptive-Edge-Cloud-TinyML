#include "adaptive_runtime.h"
#include <Arduino.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "network/mqtt_client.h"
#include "network/wifi_manager.h"
#include "meta_policy_data.h"
#include "failover_config.h"
#include "version.h"
#include "sensors/attitude_estimator.h"

namespace policy {
namespace {
constexpr const char* device = "esp32-r1";
constexpr const char* requestTopic = "gesture/esp32-r1/inference/request";
constexpr const char* responseTopic = "gesture/esp32-r1/inference/response";
constexpr const char* probeRequest = "gesture/esp32-r1/policy/probe/request";
constexpr const char* probeResponse = "gesture/esp32-r1/policy/probe/response";
constexpr const char* telemetryTopic = "gesture/esp32-r1/telemetry";
constexpr const char* telemetrySchema = "decision-r1-v1";
constexpr const char* poseTopic = "gesture/esp32-r1/pose";
constexpr uint32_t POSE_PUBLISH_INTERVAL_MS = 200;
constexpr const char* localVersion = "gesture-model-v1.1.0";
constexpr const char* cloudVersion = "gesture-full-cloud-v1.0.0";
struct Work {
    float features[10]; inference::UncertaintyResult cached;
    float localMs; uint32_t window, cachedAtMs; unsigned localCalls;
};
struct Pending {
    Work work{}; float state[6]{}; char id[96]{};
    DecisionResult result; FailoverState transition;
    bool occupied = false, waitingState = false, waitingCloud = false, done = false;
    bool controlled = false, stateValid = false;
    uint32_t startedUs = 0, rttAgeMs = 0;
    const char* failureStage = "NONE";
};
QueueHandle_t queue = nullptr;
QueueHandle_t poseQueue = nullptr;
bool workerRunning = false;
Pending slots[failover_config::PENDING_CAPACITY];
Pending* active[failover_config::PENDING_CAPACITY]{};
volatile bool wifiSnapshot = false, mqttSnapshot = false;
float lastRttMs = NAN;
uint32_t lastRttAtMs = 0, probeStartedUs = 0, lastProbeMs = 0;
bool probeActive = false, probeTimedOut = false;
char probeId[96]{};
uint32_t bootSessionNonce = 0;
attitude::PoseEstimate latestPose{};
bool poseAvailable = false;
uint32_t lastPosePublishedSequence = 0;
uint32_t lastPosePublishMs = 0;
uint32_t lastPoseLogMs = 0;

bool stringField(const char* json, const char* key, const char* value) {
    String token = String("\"") + key + "\":\"" + value + "\"";
    return strstr(json, token.c_str()) != nullptr;
}
bool numberField(const char* json, const char* key, float& value) {
    String token = String("\"") + key + "\":";
    const char* start = strstr(json, token.c_str());
    if (!start) return false;
    start += token.length(); char* end = nullptr; value = strtof(start, &end);
    return end != start && (*end == ',' || *end == '}') && std::isfinite(value);
}
void appendNumber(String& target, float value) {
    char number[32]; snprintf(number, sizeof(number), "%.9g", double(value)); target += number;
}
String array(const float* values, unsigned count) {
    String result = "[";
    for (unsigned i = 0; i < count; ++i) { if (i) result += ','; appendNumber(result, values[i]); }
    return result + ']';
}
String nullableNumber(float value, bool available) {
    if (!available || !std::isfinite(value)) return "null";
    String result;
    appendNumber(result, value);
    return result;
}
void publishPoseIfDue() {
    if (!poseAvailable || !network::isMqttConnected()) return;
    // Preserve policy-network measurements and CLOUD response priority. Pose is
    // visualization-only and must never compete with an active RTT probe/request.
    if (probeActive) return;
    for (const auto& pending : slots) {
        if (pending.occupied && pending.waitingCloud) return;
    }
    if (latestPose.sequence == 0 || latestPose.sequence == lastPosePublishedSequence) return;

    const uint32_t nowMs = millis();
    if (lastPosePublishMs != 0
        && uint32_t(nowMs - lastPosePublishMs) < POSE_PUBLISH_INTERVAL_MS) return;

    String payload = String("{\"schema_version\":\"") + attitude::POSE_SCHEMA_VERSION;
    payload += "\",\"pose_id\":\"r1-pose-";
    char identity[48];
    snprintf(identity, sizeof(identity), "%08lx-%lu",
             (unsigned long)bootSessionNonce, (unsigned long)latestPose.sequence);
    payload += identity;
    payload += "\",\"device_id\":\""; payload += device;
    payload += "\",\"timestamp_ms\":"; payload += latestPose.timestampMs;
    payload += ",\"sequence\":"; payload += latestPose.sequence;
    payload += ",\"roll_deg_est\":"; appendNumber(payload, latestPose.rollDeg);
    payload += ",\"pitch_deg_est\":"; appendNumber(payload, latestPose.pitchDeg);
    payload += ",\"yaw_rel_deg_est\":"; appendNumber(payload, latestPose.yawRelativeDeg);
    payload += ",\"estimator_version\":\""; payload += attitude::ESTIMATOR_VERSION;
    payload += "\",\"orientation_version\":\""; payload += ORIENTATION_VERSION;
    payload += "\",\"firmware_version\":\""; payload += FIRMWARE_VERSION;
    payload += "\",\"source\":\""; payload += attitude::POSE_SOURCE;
    payload += "\",\"yaw_reference\":\""; payload += attitude::YAW_REFERENCE;
    payload += "\"}";

    lastPosePublishMs = nowMs;
    if (payload.length() + strlen(poseTopic) + 7 >= network::DEFAULT_MQTT_BUFFER_BYTES) {
        Serial.printf("R1_POSE_DROPPED sequence=%lu reason=PAYLOAD_TOO_LARGE bytes=%u\n",
                      (unsigned long)latestPose.sequence, unsigned(payload.length()));
        return;
    }
    if (!network::publishMqtt(poseTopic, payload.c_str())) {
        Serial.printf("R1_POSE_DROPPED sequence=%lu reason=PUBLISH_FAILED bytes=%u\n",
                      (unsigned long)latestPose.sequence, unsigned(payload.length()));
        return;
    }

    lastPosePublishedSequence = latestPose.sequence;
    if (lastPoseLogMs == 0 || uint32_t(nowMs - lastPoseLogMs) >= 1000) {
        lastPoseLogMs = nowMs;
        Serial.printf(
            "R1_POSE {\"sequence\":%lu,\"roll_deg_est\":%.3f,\"pitch_deg_est\":%.3f,\"yaw_rel_deg_est\":%.3f,\"yaw_reference\":\"boot-relative\"}\n",
            (unsigned long)latestPose.sequence, latestPose.rollDeg, latestPose.pitchDeg, latestPose.yawRelativeDeg
        );
    }
}

void publishDecisionTelemetry(const Pending& p) {
    const auto& r = p.result;
    const auto& cached = p.work.cached;
    const char* requested = r.policy.action == 0 ? "LOCAL" : r.policy.action == 1 ? "CLOUD" : "NOT_EVALUATED";
    const char* effective = r.effectiveAction == 0 ? "LOCAL" : "CLOUD";
    const char* modelVersion = r.effectiveAction == 1 && r.success ? cloudVersion : localVersion;
    const char* rttSource = p.controlled ? "CONTROLLED" : p.stateValid ? "last_successful_probe" : "unavailable";
    const uint32_t freeHeap = p.stateValid && p.state[3] >= 0 ? uint32_t(p.state[3]) : ESP.getFreeHeap();
    const float networkMs = r.roundTripUs / 1000.f;

    String payload = String("{\"schema_version\":\"") + telemetrySchema;
    payload += "\",\"request_id\":\""; payload += p.id;
    payload += "\",\"device_id\":\""; payload += device;
    payload += "\",\"timestamp_ms\":"; payload += millis();
    payload += ",\"window_id\":"; payload += p.work.window;
    payload += ",\"requested_action\":\""; payload += requested;
    payload += "\",\"effective_action\":\""; payload += effective;
    payload += "\",\"failover\":"; payload += r.failover ? "true" : "false";
    payload += ",\"failover_reason\":\""; payload += reasonName(r.failoverReason);
    payload += "\",\"failure_stage\":\""; payload += p.failureStage;
    payload += "\",\"wifi_connected\":"; payload += r.wifiConnected ? "true" : "false";
    payload += ",\"mqtt_connected\":"; payload += r.mqttConnected ? "true" : "false";
    payload += ",\"predicted_class_id\":"; payload += r.finalPrediction;
    payload += ",\"confidence\":"; appendNumber(payload, r.finalConfidence);
    payload += ",\"uncertainty\":"; appendNumber(payload, cached.normalizedPredictiveEntropy);
    payload += ",\"rtt_ms\":"; payload += nullableNumber(p.stateValid ? p.state[5] : NAN, p.stateValid);
    payload += ",\"rtt_source\":\""; payload += rttSource;
    payload += "\",\"rtt_age_ms\":"; payload += p.rttAgeMs;
    payload += ",\"free_heap_bytes\":"; payload += freeHeap;
    payload += ",\"local_inference_ms\":"; appendNumber(payload, p.work.localMs);
    payload += ",\"request_elapsed_ms\":"; payload += nullableNumber(networkMs, r.requestIssued);
    payload += ",\"server_compute_ms\":";
    payload += nullableNumber(r.serverComputeMs, r.effectiveAction == 1 && r.success);
    payload += ",\"bytes_tx\":"; payload += r.txBytes;
    payload += ",\"bytes_rx\":"; payload += r.rxBytes;
    payload += ",\"model_version\":\""; payload += modelVersion;
    payload += "\",\"policy_version\":\""; payload += meta_policy_data::VERSION;
    payload += "\",\"firmware_version\":\""; payload += FIRMWARE_VERSION;
    payload += "\",\"success\":"; payload += r.success ? "true" : "false";
    payload += ",\"controlled\":"; payload += p.controlled ? "true" : "false";
    payload += '}';

    if (!network::isMqttConnected()) return;
    if (payload.length() + strlen(telemetryTopic) + 7 >= network::DEFAULT_MQTT_BUFFER_BYTES) {
        Serial.printf("R1_TELEMETRY_DROPPED request_id=%s reason=PAYLOAD_TOO_LARGE bytes=%u\n",
                      p.id, unsigned(payload.length()));
        return;
    }
    if (!network::publishMqtt(telemetryTopic, payload.c_str())) {
        Serial.printf("R1_TELEMETRY_DROPPED request_id=%s reason=PUBLISH_FAILED bytes=%u\n",
                      p.id, unsigned(payload.length()));
    }
}
void logDecision(const Pending& p) {
    const auto& r = p.result;
    const auto& cached = p.work.cached;
    String line = String("R1_DECISION {\"request_id\":\"") + p.id + "\",\"window_id\":" + p.work.window
        + ",\"timestamp_ms\":" + millis() + ",\"cached_at_ms\":" + p.work.cachedAtMs
        + ",\"policy_version\":\"" + meta_policy_data::VERSION + "\",\"firmware_version\":\"" + FIRMWARE_VERSION
        + "\",\"failover_config_version\":\"" + failover_config::VERSION
        + "\",\"local_model_version\":\"gesture-model-v1.1.0\",\"cloud_model_version\":\"" + cloudVersion
        + "\",\"requested_action\":\"" + (r.policy.action == 0 ? "LOCAL" : r.policy.action == 1 ? "CLOUD" : "NOT_EVALUATED")
        + "\",\"effective_action\":\"" + (r.effectiveAction == 0 ? "LOCAL" : "CLOUD")
        + "\",\"selected_action\":\"" + (r.effectiveAction == 0 ? "LOCAL" : "CLOUD")
        + "\",\"failover\":" + (r.failover ? "true" : "false")
        + ",\"failover_reason\":\"" + reasonName(r.failoverReason) + "\",\"failure_stage\":\"" + p.failureStage
        + "\",\"wifi_connected\":" + (r.wifiConnected ? "true" : "false")
        + ",\"mqtt_connected\":" + (r.mqttConnected ? "true" : "false")
        + ",\"local_prediction\":" + cached.predictedClass + ",\"final_prediction\":" + r.finalPrediction
        + ",\"confidence\":" + String(cached.confidence, 9) + ",\"entropy\":" + String(cached.predictiveEntropyNats, 9)
        + ",\"final_confidence\":" + String(r.finalConfidence, 9)
        + ",\"policy_state\":" + (p.stateValid ? array(p.state, 6) : String("null"))
        + ",\"policy_input\":" + (r.policy.action >= 0 ? array(r.policy.normalized, 6) : String("null"))
        + ",\"rtt_source\":\"" + (p.controlled ? "CONTROLLED" : p.stateValid ? "last_successful_probe" : "unavailable")
        + "\",\"rtt_age_ms\":" + p.rttAgeMs + ",\"success\":" + (r.success ? "true" : "false")
        + ",\"controlled\":" + (p.controlled ? "true" : "false")
        + ",\"local_inference_count\":" + r.localInferenceCount + ",\"second_inference_count\":" + r.secondInferenceCount
        + ",\"policy_inference_us\":" + r.policy.inferenceUs + ",\"bytes_tx\":" + r.txBytes + ",\"bytes_rx\":" + r.rxBytes
        + ",\"request_issued\":" + (r.requestIssued ? "true" : "false") + ",\"request_status\":\"" + r.requestStatus + "\"";
    if (r.requestIssued) line += ",\"request_elapsed_ms\":" + String(r.roundTripUs / 1000.f, 3);
    if (r.effectiveAction == 1 && r.success)
        line += ",\"e2e_latency_ms\":" + String(r.roundTripUs / 1000.f, 3) + ",\"server_compute_latency_ms\":" + String(r.serverComputeMs, 6);
    Serial.print(line + "}\n");
    publishDecisionTelemetry(p);
}
void finish(Pending& p) {
    auto& r = p.result;
    r.effectiveAction = p.transition.effectiveAction; r.finalPrediction = p.transition.finalClass;
    r.failover = p.transition.failover; r.failoverReason = p.transition.reason;
    r.localInferenceCount = p.work.localCalls;
    r.secondInferenceCount = p.work.localCalls > 0 ? p.work.localCalls - 1 : 0;
    r.success = p.transition.complete && r.finalPrediction >= 0 && r.finalPrediction < 5 && r.localInferenceCount == 1;
    if (r.effectiveAction == 0) r.finalConfidence = p.work.cached.confidence;
    p.waitingCloud = false; p.waitingState = false; p.done = true;
    logDecision(p);
}
void fallback(Pending& p, FailoverReason reason, const char* stage) {
    p.failureStage = stage;
    if (p.result.requestIssued) p.result.roundTripUs = micros() - p.startedUs;
    p.transition.fallback(reason); finish(p);
}
void updateConnectivity(Pending& p) {
    p.result.wifiConnected = network::isWifiConnected();
    p.result.mqttConnected = p.result.wifiConnected && network::isMqttConnected();
}
void onResponse(const char* topic, const uint8_t* payload, unsigned length) {
    if (length >= 1024) return;
    char response[1024]; memcpy(response, payload, length); response[length] = '\0';
    if (strcmp(topic, probeResponse) == 0 && probeActive
        && stringField(response, "request_id", probeId) && stringField(response, "schema_version", "r1-probe-v1")
        && strstr(response, "\"probe\":true")) {
        lastRttMs = (micros() - probeStartedUs) / 1000.f; lastRttAtMs = millis();
        probeActive = false; probeTimedOut = false; return;
    }
    if (strcmp(topic, responseTopic)) return;
    for (auto* p : active) {
        if (!p || !p->waitingCloud || !stringField(response, "request_id", p->id)) continue;
        float predicted, confidence, serverMs;
        if (!stringField(response, "schema_version", "inference-r1-v1") || !stringField(response, "mode", "CLOUD")
            || !strstr(response, "\"success\":true") || !stringField(response, "model_version", cloudVersion)
            || !stringField(response, "policy_version", meta_policy_data::VERSION)
            || !numberField(response, "predicted_class_id", predicted) || predicted < 0 || predicted > 4 || floorf(predicted) != predicted
            || !numberField(response, "confidence", confidence) || confidence < 0 || confidence > 1
            || !numberField(response, "server_latency_ms", serverMs) || serverMs < 0) return;
        // Deadline wins even if a late message is delivered before the next timer poll.
        if (requestExpired(micros(), p->startedUs, failover_config::CLOUD_RESPONSE_TIMEOUT_MS)) return;
        p->result.roundTripUs = micros() - p->startedUs;
        p->result.rxBytes = length; p->result.serverComputeMs = serverMs;
        p->result.finalConfidence = confidence; p->result.requestStatus = "SUCCESS";
        p->transition.cloudSuccess(int(predicted)); finish(*p); return;
    }
}
void begin(Pending& p) {
    updateConnectivity(p);
    if (!runMeta(p.state, p.result.policy)) {
        p.transition.begin(-1, p.result.wifiConnected, p.result.mqttConnected, p.work.cached.predictedClass, p.work.localCalls);
        fallback(p, FailoverReason::LOCAL_QUEUE_SAFETY, "POLICY_INPUT_INVALID"); return;
    }
    if (!p.transition.begin(p.result.policy.action, p.result.wifiConnected, p.result.mqttConnected,
                            p.work.cached.predictedClass, p.work.localCalls)) { p.done = true; return; }
    if (p.transition.complete) { p.failureStage = p.transition.failover ? "CONNECTIVITY_GATE" : "NONE"; finish(p); return; }
    String payload = String("{\"schema_version\":\"inference-r1-v1\",\"request_id\":\"") + p.id
        + "\",\"device_id\":\"" + device + "\",\"timestamp_ms\":" + millis()
        + ",\"mode\":\"CLOUD\",\"features\":" + array(p.work.features, 10)
        + ",\"feature_version\":\"features-v1\",\"feature_encoding\":\"normalized-float32\""
        + ",\"model_version\":\"" + cloudVersion + "\",\"policy_version\":\"" + meta_policy_data::VERSION
        + "\",\"firmware_version\":\"" + FIRMWARE_VERSION + "\",\"confidence\":";
    appendNumber(payload, p.state[0]); payload += ",\"entropy\":"; appendNumber(payload, p.state[1]);
    payload += ",\"margin\":"; appendNumber(payload, p.state[2]); payload += ",\"policy_state\":" + array(p.state, 6) + '}';
    p.startedUs = micros();
    if (payload.length() + strlen(requestTopic) + 7 >= network::DEFAULT_MQTT_BUFFER_BYTES
        || !network::publishMqtt(requestTopic, payload.c_str())) {
        p.result.requestStatus = "PUBLISH_FAILED"; fallback(p, FailoverReason::CLOUD_PUBLISH_FAILED, "PUBLISH"); return;
    }
    p.result.txBytes = payload.length(); p.result.requestIssued = true;
    p.result.requestStatus = "PENDING"; p.waitingCloud = true;
}
void poll(Pending& p) {
    if (!p.waitingCloud) return;
    updateConnectivity(p);
    if (!p.result.wifiConnected) { p.result.requestStatus = "DISCONNECTED"; fallback(p, FailoverReason::WIFI_UNAVAILABLE, "WAIT_RESPONSE"); }
    else if (!p.result.mqttConnected) { p.result.requestStatus = "DISCONNECTED"; fallback(p, FailoverReason::MQTT_UNAVAILABLE, "WAIT_RESPONSE"); }
    else if (requestExpired(micros(), p.startedUs, failover_config::CLOUD_RESPONSE_TIMEOUT_MS)) {
        p.result.requestStatus = "TIMEOUT"; fallback(p, FailoverReason::CLOUD_RESPONSE_TIMEOUT, "WAIT_RESPONSE");
    }
}
void prepareState(Pending& p) {
    updateConnectivity(p);
    if (!std::isfinite(lastRttMs)) {
        p.transition.begin(-1, p.result.wifiConnected, p.result.mqttConnected, p.work.cached.predictedClass, p.work.localCalls);
        if (p.transition.complete) { p.failureStage = "PRE_POLICY_CONNECTIVITY"; finish(p); }
        else if (probeTimedOut) fallback(p, FailoverReason::CLOUD_RESPONSE_TIMEOUT, "PRE_POLICY_RTT_PROBE");
        else p.waitingState = true;
        return;
    }
    float first = 0, second = 0;
    for (float v : p.work.cached.meanProbabilities) { if (v > first) { second = first; first = v; } else if (v > second) second = v; }
    const float state[6] = {p.work.cached.confidence, p.work.cached.predictiveEntropyNats, first-second,
        float(ESP.getFreeHeap()), p.work.localMs, lastRttMs};
    memcpy(p.state, state, sizeof(state)); p.stateValid = true; p.rttAgeMs = millis() - lastRttAtMs;
    p.waitingState = false; begin(p);
}
void worker(void*) {
    uint32_t lastRetryMs = 0;
    bool firstConnect = true, oldWifi = false, oldMqtt = false;
    for (;;) {
        bool wifi = network::isWifiConnected(), mqtt = wifi && network::isMqttConnected();
        if (wifi != oldWifi || mqtt != oldMqtt) {
            Serial.printf("R1_CONNECTIVITY {\"timestamp_ms\":%lu,\"wifi_connected\":%s,\"mqtt_connected\":%s}\n",
                (unsigned long)millis(), wifi ? "true" : "false", mqtt ? "true" : "false");
            // Back off after loss; do not hold the inference queue during reconnect.
            if (!wifi || !mqtt) lastRetryMs = millis();
            oldWifi = wifi; oldMqtt = mqtt;
        }
        wifiSnapshot = wifi; mqttSnapshot = mqtt;
        if ((!wifi || !mqtt) && (firstConnect || uint32_t(millis()-lastRetryMs) >= failover_config::RECONNECT_INTERVAL_MS)) {
            firstConnect = false; lastRetryMs = millis();
            if (!wifi) network::connectWifi(0); // Existing helper starts association without its 20s wait.
            if (network::isWifiConnected()) connectAdaptiveNetwork();
        }
        network::mqttLoop();
        attitude::PoseEstimate poseUpdate;
        while (poseQueue && xQueueReceive(poseQueue, &poseUpdate, 0) == pdTRUE) {
            latestPose = poseUpdate;
            poseAvailable = latestPose.sequence > 0;
        }
        if (probeActive && requestExpired(micros(), probeStartedUs, failover_config::RTT_PROBE_TIMEOUT_MS)) {
            probeActive = false; probeTimedOut = true;
        }
        if (network::isWifiConnected() && network::isMqttConnected() && !probeActive
            && (lastProbeMs == 0 || uint32_t(millis()-lastProbeMs) >= failover_config::RTT_PROBE_INTERVAL_MS)) {
            lastProbeMs = millis();
            snprintf(probeId, sizeof(probeId), "r1-probe-%08lx-%lu",
                     (unsigned long)bootSessionNonce, (unsigned long)lastProbeMs);
            String payload = String("{\"schema_version\":\"r1-probe-v1\",\"request_id\":\"") + probeId + "\",\"device_id\":\"" + device + "\"}";
            probeStartedUs = micros(); probeActive = network::publishMqtt(probeRequest, payload.c_str());
            if (!probeActive) probeTimedOut = true;
        }
        for (unsigned i=0; i<failover_config::PENDING_CAPACITY; ++i) {
            auto& p = slots[i];
            if (!p.occupied) continue;
            if (p.waitingState) prepareState(p);
            poll(p);
            if (p.done) { active[i] = nullptr; p.occupied = false; }
        }
        Work work;
        if (xQueueReceive(queue, &work, 0) == pdTRUE) {
            unsigned slot = 0; while (slot < failover_config::PENDING_CAPACITY && slots[slot].occupied) ++slot;
            if (slot < failover_config::PENDING_CAPACITY) {
                auto& p = slots[slot]; p = Pending{}; p.work = work; p.occupied = true;
                snprintf(p.id, sizeof(p.id), "r1-%08lx-%lu-%lu",
                         (unsigned long)bootSessionNonce,
                         (unsigned long)work.cachedAtMs,
                         (unsigned long)work.window);
                active[slot] = &p; prepareState(p);
            } else {
                Pending p; p.work = work;
                snprintf(p.id, sizeof(p.id), "r1-capacity-%08lx-%lu",
                         (unsigned long)bootSessionNonce, (unsigned long)work.window);
                updateConnectivity(p); p.transition.begin(-1, true, true, work.cached.predictedClass, work.localCalls);
                fallback(p, FailoverReason::LOCAL_QUEUE_SAFETY, "PENDING_CAPACITY");
            }
        }
        // Pose is auxiliary dashboard telemetry, so publish it only after policy,
        // failover, cloud-response and work-queue handling for this iteration.
        publishPoseIfDue();
        delay(1);
    }
}
}

bool connectAdaptiveNetwork() {
    if (!network::isWifiConnected() && !network::connectWifi()) return false;
    if (network::isMqttConnected()) return true;
    if (!network::configureMqtt("192.168.137.1", 1883)) return false;
    network::setMqttSocketTimeoutSeconds(failover_config::MQTT_SOCKET_TIMEOUT_SECONDS);
    return network::setMqttMessageHandler(onResponse) && network::connectMqtt(device)
        && network::subscribeMqtt(responseTopic) && network::subscribeMqtt(probeResponse);
}

bool processCachedDecision(const float features[10], const inference::UncertaintyResult& cached,
                          const float state[6], const char* id, uint32_t window,
                          DecisionResult& result, bool controlled, unsigned localCalls) {
    if (workerRunning || !features || !state || !id || strlen(id) >= 96 || localCalls != 1) return false;
    for (unsigned i=0; i<10; ++i) if (!std::isfinite(features[i])) return false;
    Pending p; memcpy(p.work.features, features, sizeof(p.work.features)); p.work.cached = cached;
    p.work.window = window; p.work.cachedAtMs = millis(); p.work.localCalls = localCalls;
    memcpy(p.state, state, sizeof(p.state)); snprintf(p.id, sizeof(p.id), "%s", id);
    p.stateValid = true; p.controlled = controlled; active[0] = &p;
    begin(p);
    while (!p.done) { network::mqttLoop(); poll(p); delay(1); }
    active[0] = nullptr; result = p.result; return result.success;
}

bool startAdaptiveRuntime() {
    if (queue) return true;
    if (!initializeMeta()) return false;
    // Request IDs are persisted under a (device_id, request_id) uniqueness constraint.
    // millis()/window counters restart after every reboot, so add a per-boot nonce to
    // prevent normal device restarts from colliding with prior persisted sessions.
    bootSessionNonce = esp_random();
    if (bootSessionNonce == 0) bootSessionNonce = uint32_t(micros()) ^ 0xA5C31F27u;
    Serial.printf("R1_SESSION {\"boot_nonce\":\"%08lx\"}\n", (unsigned long)bootSessionNonce);
    queue = xQueueCreate(failover_config::WORK_QUEUE_CAPACITY, sizeof(Work));
    if (!queue) return false;
    poseQueue = xQueueCreate(1, sizeof(attitude::PoseEstimate));
    if (!poseQueue) {
        vQueueDelete(queue); queue = nullptr;
        return false;
    }
    latestPose = attitude::PoseEstimate{};
    poseAvailable = false;
    lastPosePublishedSequence = 0;
    lastPosePublishMs = 0;
    lastPoseLogMs = 0;
    workerRunning = true;
    if (xTaskCreatePinnedToCore(worker, "r1-policy", 12288, nullptr, 1, nullptr, 0) != pdPASS) {
        vQueueDelete(poseQueue); poseQueue = nullptr;
        vQueueDelete(queue); queue = nullptr; workerRunning = false; return false;
    }
    return true;
}
bool submitCachedDecision(const float features[10], const inference::UncertaintyResult& cached,
                          float localMs, uint32_t window, unsigned localCalls) {
    if (!queue || localCalls != 1) return false;
    Work work; memcpy(work.features, features, sizeof(work.features)); work.cached = cached;
    work.localMs = localMs; work.window = window; work.cachedAtMs = millis(); work.localCalls = localCalls;
    if (xQueueSend(queue, &work, 0) == pdTRUE) return true;
    // Defensive result delivery even if an unexpected scheduler stall exhausts capacity.
    Pending p; p.work = work;
    snprintf(p.id, sizeof(p.id), "r1-queue-%08lx-%lu",
             (unsigned long)bootSessionNonce, (unsigned long)window);
    p.result.wifiConnected = wifiSnapshot; p.result.mqttConnected = mqttSnapshot;
    p.transition.begin(-1, true, true, cached.predictedClass, localCalls);
    fallback(p, FailoverReason::LOCAL_QUEUE_SAFETY, "WORK_QUEUE");
    return p.result.success;
}

bool submitPoseEstimate(const attitude::PoseEstimate& pose) {
    if (!poseQueue || pose.sequence == 0
        || !std::isfinite(pose.rollDeg)
        || !std::isfinite(pose.pitchDeg)
        || !std::isfinite(pose.yawRelativeDeg)) return false;
    // Queue length is exactly one: overwrite keeps only the freshest 100 Hz pose
    // without ever blocking the sampling/inference loop.
    return xQueueOverwrite(poseQueue, &pose) == pdPASS;
}
}
