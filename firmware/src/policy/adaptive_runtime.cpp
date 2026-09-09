#include "adaptive_runtime.h"
#include <Arduino.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "network/mqtt_client.h"
#include "network/wifi_manager.h"
#include "meta_policy_data.h"

namespace policy {
namespace {
constexpr const char* device = "esp32-r1";
constexpr const char* requestTopic = "gesture/esp32-r1/inference/request";
constexpr const char* responseTopic = "gesture/esp32-r1/inference/response";
constexpr const char* probeRequest = "gesture/esp32-r1/policy/probe/request";
constexpr const char* probeResponse = "gesture/esp32-r1/policy/probe/response";
constexpr const char* cloudVersion = "gesture-full-cloud-v1.0.0";
constexpr const char* firmwareVersion = "0.2.0-r1";
char response[1024], expectedId[96];
bool received = false, probing = false;
uint32_t receivedUs;
unsigned responseBytes;
QueueHandle_t queue = nullptr;
struct Work { float features[10]; inference::UncertaintyResult cached; float localMs; uint32_t window; };

bool stringField(const char* json, const char* key, const char* value) {
    String token = String("\"") + key + "\":\"" + value + "\"";
    return strstr(json, token.c_str()) != nullptr;
}
bool numberField(const char* json, const char* key, float& value) {
    String token = String("\"") + key + "\":";
    const char* start = strstr(json, token.c_str());
    if (!start) return false;
    start += token.length();
    char* end = nullptr;
    value = strtof(start, &end);
    return end != start && (*end == ',' || *end == '}') && std::isfinite(value);
}
void onResponse(const char* topic, const uint8_t* payload, unsigned length) {
    if (strcmp(topic, probing ? probeResponse : responseTopic) || length >= sizeof(response)) return;
    memcpy(response, payload, length); response[length] = '\0';
    if (!stringField(response, "request_id", expectedId)) return;
    if (!stringField(response, "schema_version", probing ? "r1-probe-v1" : "inference-r1-v1")) return;
    receivedUs = micros(); responseBytes = length; received = true;
}
bool exchange(const char* topic, const String& payload, const char* id, bool probe, uint32_t& elapsed,
              unsigned* publishedBytes = nullptr) {
    if (payload.length() + strlen(topic) + 7 >= network::DEFAULT_MQTT_BUFFER_BYTES) return false;
    snprintf(expectedId, sizeof(expectedId), "%s", id);
    probing = probe; received = false;
    const uint32_t started = micros();
    if (!network::publishMqtt(topic, payload.c_str())) return false;
    if (publishedBytes) *publishedBytes = payload.length();
    while (!received && uint32_t(micros() - started) < 3000000UL) {
        network::mqttLoop(); delay(1);
    }
    if (received) elapsed = receivedUs - started;
    return received;
}
void appendNumber(String& target, float value) {
    char number[32]; snprintf(number, sizeof(number), "%.9g", double(value)); target += number;
}
String array(const float* values, unsigned count) {
    String result = "[";
    for (unsigned i = 0; i < count; ++i) { if (i) result += ','; appendNumber(result, values[i]); }
    return result + ']';
}
void logDecision(const float state[6], const inference::UncertaintyResult& cached, const char* id,
                 uint32_t window, const DecisionResult& result, bool controlled) {
    String line = String("R1_DECISION {\"request_id\":\"") + id + "\",\"window_id\":" + window
        + ",\"timestamp_ms\":" + millis() + ",\"policy_version\":\"" + meta_policy_data::VERSION
        + "\",\"firmware_version\":\"" + firmwareVersion + "\",\"local_model_version\":\"gesture-model-v1.1.0\""
        + ",\"cloud_model_version\":\"" + cloudVersion + "\",\"selected_action\":\""
        + (result.policy.action == 0 ? "LOCAL" : result.policy.action == 1 ? "CLOUD" : "INVALID")
        + "\",\"local_prediction\":" + cached.predictedClass + ",\"final_prediction\":" + result.finalPrediction
        + ",\"policy_state\":" + array(state, 6) + ",\"policy_input\":" + array(result.policy.normalized, 6)
        + ",\"confidence\":" + String(state[0], 9) + ",\"entropy\":" + String(state[1], 9)
        + ",\"margin\":" + String(state[2], 9) + ",\"success\":" + (result.success ? "true" : "false")
        + ",\"controlled\":" + (controlled ? "true" : "false") + ",\"second_inference_count\":0"
        + ",\"policy_inference_us\":" + result.policy.inferenceUs
        + ",\"bytes_tx\":" + result.txBytes + ",\"bytes_rx\":" + result.rxBytes;
    if (result.policy.action == 1 && result.success) {
        line += ",\"e2e_latency_ms\":" + String(result.roundTripUs / 1000.0f, 3)
            + ",\"server_compute_latency_ms\":" + String(result.serverComputeMs, 6);
    }
    Serial.print(line + "}\n");
}
void worker(void*) {
    Work work;
    for (;;) {
        if (xQueueReceive(queue, &work, portMAX_DELAY) != pdTRUE) continue;
        char id[80]; snprintf(id, sizeof(id), "r1-%lu-%lu", (unsigned long)millis(), (unsigned long)work.window);
        if (!connectAdaptiveNetwork()) { Serial.printf("R1_STATE_UNAVAILABLE window=%lu network=disconnected\n", (unsigned long)work.window); continue; }
        const String payload = String("{\"schema_version\":\"r1-probe-v1\",\"request_id\":\"") + id
            + "\",\"device_id\":\"" + device + "\"}";
        uint32_t probeUs = 0;
        if (!exchange(probeRequest, payload, id, true, probeUs)) {
            Serial.printf("R1_STATE_UNAVAILABLE window=%lu probe=timeout\n", (unsigned long)work.window); continue;
        }
        float first = 0, second = 0;
        for (float p : work.cached.meanProbabilities) { if (p > first) { second = first; first = p; } else if (p > second) second = p; }
        const float state[6] = {work.cached.confidence, work.cached.predictiveEntropyNats, first - second,
            float(ESP.getFreeHeap()), work.localMs, probeUs / 1000.0f};
        DecisionResult result;
        processCachedDecision(work.features, work.cached, state, id, work.window, result);
    }
}
}

bool connectAdaptiveNetwork() {
    if (!network::isWifiConnected() && !network::connectWifi()) return false;
    if (network::isMqttConnected()) return true;
    return network::configureMqtt("192.168.137.1", 1883)
        && network::setMqttMessageHandler(onResponse) && network::connectMqtt(device)
        && network::subscribeMqtt(responseTopic) && network::subscribeMqtt(probeResponse);
}

bool processCachedDecision(const float features[10], const inference::UncertaintyResult& cached,
                          const float state[6], const char* id, uint32_t window,
                          DecisionResult& result, bool controlled) {
    result = DecisionResult{};
    if (!features || !state || !id || strlen(id) >= sizeof(expectedId)) return false;
    for (unsigned i = 0; i < 10; ++i) if (!std::isfinite(features[i])) return false;
    if (!runMeta(state, result.policy)) return false;
    if (result.policy.action == 0) {
        result.finalPrediction = cached.predictedClass;
        result.success = cached.predictedClass >= 0 && cached.predictedClass < 5;
    } else {
        String payload = String("{\"schema_version\":\"inference-r1-v1\",\"request_id\":\"") + id
            + "\",\"device_id\":\"" + device + "\",\"timestamp_ms\":" + millis()
            + ",\"mode\":\"CLOUD\",\"features\":" + array(features, 10)
            + ",\"feature_version\":\"features-v1\",\"feature_encoding\":\"normalized-float32\""
            + ",\"model_version\":\"" + cloudVersion + "\",\"policy_version\":\"" + meta_policy_data::VERSION
            + "\",\"firmware_version\":\"" + firmwareVersion + "\",\"confidence\":";
        appendNumber(payload, state[0]); payload += ",\"entropy\":"; appendNumber(payload, state[1]);
        payload += ",\"margin\":"; appendNumber(payload, state[2]); payload += ",\"policy_state\":" + array(state, 6) + '}';
        // Application payload bytes; MQTT/TCP/Wi-Fi framing is not included.
        if (exchange(requestTopic, payload, id, false, result.roundTripUs, &result.txBytes)) {
            result.rxBytes = responseBytes;
            float prediction, confidence;
            if (stringField(response, "mode", "CLOUD") && strstr(response, "\"success\":true")
                && stringField(response, "model_version", cloudVersion)
                && stringField(response, "policy_version", meta_policy_data::VERSION)
                && numberField(response, "predicted_class_id", prediction) && prediction >= 0 && prediction < 5
                && prediction == floorf(prediction) && numberField(response, "confidence", confidence)
                && confidence >= 0 && confidence <= 1
                && numberField(response, "server_latency_ms", result.serverComputeMs) && result.serverComputeMs >= 0) {
                result.finalPrediction = int(prediction); result.success = true;
            }
        }
    }
    logDecision(state, cached, id, window, result, controlled);
    return result.success;
}

bool startAdaptiveRuntime() {
    if (queue) return true;
    if (!initializeMeta()) return false;
    queue = xQueueCreate(2, sizeof(Work));
    if (!queue) return false;
    if (xTaskCreatePinnedToCore(worker, "r1-policy", 12288, nullptr, 1, nullptr, 0) != pdPASS) {
        vQueueDelete(queue); queue = nullptr; return false;
    }
    return true;
}
bool submitCachedDecision(const float features[10], const inference::UncertaintyResult& cached,
                          float localMs, uint32_t window) {
    if (!queue) return false;
    Work work; memcpy(work.features, features, sizeof(work.features));
    work.cached = cached; work.localMs = localMs; work.window = window;
    if (xQueueSend(queue, &work, 0) == pdTRUE) return true;
    Serial.printf("R1_QUEUE_FULL window=%lu cached_local=%d\n", (unsigned long)window, cached.predictedClass);
    return false;
}
}
