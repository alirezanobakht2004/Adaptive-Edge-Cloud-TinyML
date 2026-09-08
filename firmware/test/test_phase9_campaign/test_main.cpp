// Isolated host-commanded benchmark image. No production inference source changes.
#include <Arduino.h>
#include <unity.h>
#include <cmath>
#include <cstring>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "split1_model_data.h"
#include "split2_model_data.h"
#include "prefix_runner.h"
#include "input_preprocessor.h"
#include "uncertainty.h"
#include "network/mqtt_client.h"
#include "network/wifi_manager.h"
#include "../test_phase7_split1_parity/split1_parity_vectors.h"
#include "../test_phase7_split2_parity/split2_parity_vectors.h"

namespace {
alignas(16) uint8_t arena[64 * 1024];
const char* requestTopic = "phase9/esp32-policy/request";
const char* responseTopic = "phase9/esp32-policy/response";
char response[512];
char expectedId[96];
bool received = false;
unsigned responseBytes = 0;
uint32_t responseUs = 0;

void callback(const char* topic, const uint8_t* payload, unsigned length) {
    if (strcmp(topic, responseTopic) || length >= sizeof(response)) return;
    memcpy(response, payload, length); response[length] = 0;
    if (!strstr(response, expectedId)) return;
    received = true; responseBytes = length; responseUs = micros();
}

bool prefix(int split, const float* input, float* output) {
    static tflite::MicroMutableOpResolver<1> resolver;
    static bool ready = false;
    if (!ready) { if (resolver.AddFullyConnected() != kTfLiteOk) return false; ready = true; }
    const auto* model = tflite::GetModel(split == 1 ? split1_model_data::MODEL : split2_model_data::MODEL);
    tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
    if (interpreter.AllocateTensors() != kTfLiteOk) return false;
    memcpy(interpreter.input(0)->data.f, input, 10 * sizeof(float));
    if (interpreter.Invoke() != kTfLiteOk) return false;
    memcpy(output, interpreter.output(0)->data.f, (split == 1 ? 64 : 48) * sizeof(float));
    return true;
}

bool local(const float* input, inference::UncertaintyResult& result) {
    float embedding[32];
    return inference::runPrefixB3(input, embedding) && inference::runStochasticUncertaintyFromEmbedding(embedding, result);
}

void testCampaignPrefixParity() {
    TEST_ASSERT_TRUE(inference::initPrefixRunner());
    for (int split = 1; split <= 2; ++split) {
        float actual[64];
        TEST_ASSERT_TRUE(prefix(split, split1_parity_vectors::NORMALIZED_INPUTS[0], actual));
        for (int i = 0; i < (split == 1 ? 64 : 48); ++i) {
            const float expected = split == 1 ? split1_parity_vectors::EXPECTED_OUTPUTS[0][i] : split2_parity_vectors::EXPECTED_OUTPUTS[0][i];
            TEST_ASSERT_FLOAT_WITHIN(1e-5f, expected, actual[i]);
        }
    }
}

bool exchange(const String& payload, const String& id, uint32_t& elapsed, uint32_t& sent) {
    snprintf(expectedId, sizeof(expectedId), "\"request_id\":\"%s\"", id.c_str());
    received = false; responseBytes = 0;
    sent = micros();
    if (payload.length() + strlen(requestTopic) + 7 >= network::DEFAULT_MQTT_BUFFER_BYTES ||
        !network::publishMqtt(requestTopic, payload.c_str())) { elapsed = micros() - sent; return false; }
    while (!received && uint32_t(micros() - sent) < 10000000u) { network::mqttLoop(); delay(1); }
    elapsed = micros() - sent;
    return received;
}

double field(const char* key, double fallback) {
    String needle = String("\"") + key + "\":";
    const char* location = strstr(response, needle.c_str());
    return location ? atof(location + needle.length()) : fallback;
}

void campaign(String command) {
    char buffer[384]; command.toCharArray(buffer, sizeof(buffer));
    char* token = strtok(buffer, " ");
    token = strtok(nullptr, " "); if (!token) return; unsigned window = strtoul(token, nullptr, 10);
    token = strtok(nullptr, " "); if (!token) return; int action = atoi(token);
    if (action < 0 || action > 3) return;
    float raw[10], normalized[10];
    for (int i = 0; i < 10; ++i) { token = strtok(nullptr, " "); if (!token) return; raw[i] = atof(token); if (!std::isfinite(raw[i])) return; }
    uint32_t totalStart = micros();
    uint32_t started = micros(); if (!inference::normalizeFeaturesV1(raw, normalized)) return;
    uint32_t prepUs = micros() - started;
    unsigned seed = 42000u + window;
    inference::seedUncertaintyMaskPrng(seed);
    inference::UncertaintyResult state;
    started = micros(); if (!local(normalized, state)) return; uint32_t stateUs = micros() - started;
    float first = 0, second = 0;
    for (float value : state.meanProbabilities) { if (value >= first) { second = first; first = value; } else if (value > second) second = value; }
    if (!network::isMqttConnected()) { network::connectMqtt("esp32-policy-campaign"); network::subscribeMqtt(responseTopic); }
    String id = "p9-" + String(millis()) + "-" + String(window);
    String probe = "{\"request_id\":\"" + id + "-probe\",\"probe\":true}";
    uint32_t probeUs = 0, probeSent = 0;
    bool probeOk = exchange(probe, id + "-probe", probeUs, probeSent);
    unsigned probeResponseBytes = responseBytes;
    bool connected = network::isMqttConnected();
    unsigned heap = ESP.getFreeHeap(), psram = ESP.getPsramSize(), psramFree = ESP.getFreePsram();
    uint32_t prefixUs = 0, roundtripUs = 0, sentUs = 0, receiveUs = 0;
    unsigned requestBytes = 0, resultBytes = 0;
    int prediction = -1; float confidence = -1, cloudMs = -1;
    bool success = false;
    started = micros();
    if (action == 0) {
        inference::UncertaintyResult result;
        inference::seedUncertaintyMaskPrng(seed);
        success = local(normalized, result);
        if (success) { prediction = result.predictedClass; confidence = result.confidence; }
    } else {
        float embedding[64] = {}; int count = action == 1 ? 64 : action == 2 ? 48 : 10;
        uint32_t prefixStart = micros();
        bool ok = true;
        if (action < 3) ok = prefix(action, normalized, embedding);
        else memcpy(embedding, normalized, 10 * sizeof(float));
        prefixUs = action < 3 ? micros() - prefixStart : 0;
        String payload = "{\"request_id\":\"" + id + "\",\"action\":" + String(action) + ",\"values\":[";
        for (int i = 0; i < count; ++i) { char number[32]; snprintf(number, sizeof(number), "%.9g", double(embedding[i])); if (i) payload += ','; payload += number; }
        payload += "],\"model_version\":\"gesture-model-v1.1.0\"}";
        requestBytes = payload.length();
        if (ok && exchange(payload, id, roundtripUs, sentUs)) {
            resultBytes = responseBytes; receiveUs = responseUs;
            prediction = int(field("predicted_class_id", -1)); confidence = field("confidence", -1); cloudMs = field("server_latency_ms", -1);
            success = int(field("action", -1)) == action && prediction >= 0 && prediction < 5 && confidence >= 0 && confidence <= 1 && cloudMs >= 0;
            const char* version = action == 1 ? "gesture-cloud-tail-split1-v1.0.0" : action == 2 ? "gesture-cloud-tail-split2-v1.0.0" : "gesture-full-cloud-v1.0.0";
            success = success && strstr(response, (String("\"model_version\":\"") + version + "\"").c_str());
        }
    }
    uint32_t actionUs = micros() - started, totalUs = micros() - totalStart;
    Serial.printf("POLICY_TRACE {\"trace_version\":\"esp32-decision-trace-v1\",\"window_id\":%u,\"action\":%d,", window, action);
    Serial.printf("\"request_id\":\"%s\",", id.c_str());
    Serial.printf("\"confidence\":%.9g,\"entropy\":%.9g,\"margin\":%.9g,\"state_class\":%d,", double(state.confidence), double(state.predictiveEntropyNats), double(first-second), state.predictedClass);
    Serial.printf("\"free_heap\":%u,\"psram_total\":%u,\"psram_free\":%u,\"arena_used\":%u,\"cpu_mhz\":%u,\"state_us\":%u,\"prep_us\":%u,", heap, psram, psramFree, unsigned(inference::prefixRunnerTensorArenaUsedBytes()), getCpuFrequencyMhz(), stateUs, prepUs);
    Serial.printf("\"probe_us\":%s,\"connected\":%s,\"rssi_dbm\":%d,", probeOk ? String(probeUs).c_str() : "null", connected ? "true" : "false", int(network::wifiRssiDbm()));
    Serial.printf("\"prefix_us\":%u,\"action_us\":%u,\"total_us\":%u,\"request_bytes\":%u,\"response_bytes\":%u,", prefixUs, actionUs, totalUs, requestBytes, resultBytes);
    Serial.printf("\"roundtrip_us\":%s,\"send_us\":%s,\"receive_us\":%s,\"cloud_ms\":%s,", action ? String(roundtripUs).c_str() : "null", action ? String(sentUs).c_str() : "null", receiveUs ? String(receiveUs).c_str() : "null", cloudMs >= 0 ? String(cloudMs, 6).c_str() : "null");
    Serial.printf("\"prediction\":%s,\"result_confidence\":%s,\"status\":\"%s\",\"seed\":%u,\"probe_request_bytes\":%u,\"probe_response_bytes\":%u}\n", success ? String(prediction).c_str() : "null", success ? String(confidence, 9).c_str() : "null", success ? "success" : "timeout", seed, probe.length(), probeResponseBytes);
}
}

void setUp() {}
void tearDown() {}
void setup() {
    Serial.begin(115200); delay(2000); UNITY_BEGIN(); RUN_TEST(testCampaignPrefixParity); UNITY_END();
    // PlatformIO Unity calls Serial.end(); the interactive campaign needs it again.
    Serial.begin(115200);
    network::connectWifi(); network::configureMqtt("192.168.137.1", 1883); network::setMqttMessageHandler(callback);
    network::connectMqtt("esp32-policy-campaign"); network::subscribeMqtt(responseTopic);
    Serial.setTimeout(1000); Serial.println("POLICY_CAMPAIGN_READY");
}
void loop() {
    network::mqttLoop();
    if (Serial.available()) { String line = Serial.readStringUntil('\n'); line.trim(); if (line == "PING") Serial.println("POLICY_CAMPAIGN_READY"); else if (line.startsWith("RUN ")) campaign(line); }
    delay(1);
}
