// Isolated Split1 integration test; production inference is unchanged.
#include <Arduino.h>
#include <unity.h>
#include <cmath>
#include <cstring>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "split1_model_data.h"
#include "../test_phase7_split1_parity/split1_parity_vectors.h"
#include "network/mqtt_client.h"
#include "network/wifi_manager.h"
#include "cloud_expectations.h"

namespace {
alignas(16) uint8_t arena[64 * 1024];
constexpr const char* requestTopic = "gesture/esp32-split1-test/inference/request";
constexpr const char* responseTopic = "gesture/esp32-split1-test/inference/response";
char response[512];
char expectedRequest[80];
bool received = false;

void onResponse(const char* topic, const uint8_t* payload, unsigned int length) {
    if (strcmp(topic, responseTopic) != 0 || length >= sizeof(response)) return;
    memcpy(response, payload, length);
    response[length] = '\0';
    if (strstr(response, expectedRequest) != nullptr) received = true;
}

void testSplit1EndToEnd() {
    static tflite::MicroMutableOpResolver<1> resolver;
    TEST_ASSERT_EQUAL(kTfLiteOk, resolver.AddFullyConnected());
    static tflite::MicroInterpreter interpreter(
        tflite::GetModel(split1_model_data::MODEL), resolver, arena, sizeof(arena));
    TEST_ASSERT_EQUAL(kTfLiteOk, interpreter.AllocateTensors());
    TEST_ASSERT_TRUE(network::connectWifi());
    TEST_ASSERT_TRUE(network::configureMqtt("192.168.137.1", 1883));
    TEST_ASSERT_TRUE(network::setMqttMessageHandler(onResponse));
    TEST_ASSERT_TRUE(network::connectMqtt("esp32-split1-e2e"));
    TEST_ASSERT_TRUE(network::subscribeMqtt(responseTopic));

    for (size_t v = 0; v < split1_parity_vectors::VECTOR_COUNT; ++v) {
        memcpy(interpreter.input(0)->data.f, split1_parity_vectors::NORMALIZED_INPUTS[v],
               10 * sizeof(float));
        TEST_ASSERT_EQUAL(kTfLiteOk, interpreter.Invoke());
        String payload = "{\"request_id\":\"split1-e2e-" + String(v) +
            "\",\"device_id\":\"esp32-split1-test\",\"timestamp_ms\":" + String(millis()) +
            ",\"split\":1,\"model_version\":\"gesture-model-v1.1.0\",\"embedding\":[";
        for (size_t i = 0; i < 64; ++i) {
            const float value = interpreter.output(0)->data.f[i];
            TEST_ASSERT_TRUE(std::isfinite(value));
            TEST_ASSERT_FLOAT_WITHIN(2e-5f, split1_parity_vectors::EXPECTED_OUTPUTS[v][i], value);
            char number[32];
            snprintf(number, sizeof(number), "%.9g", static_cast<double>(value));
            if (i) payload += ',';
            payload += number;
        }
        payload += "]}";
        // MQTT header allowance + topic length + payload must fit existing buffer.
        TEST_ASSERT_TRUE(payload.length() + strlen(requestTopic) + 7 < network::DEFAULT_MQTT_BUFFER_BYTES);
        snprintf(expectedRequest, sizeof(expectedRequest), "\"request_id\":\"split1-e2e-%u\"", unsigned(v));
        received = false;
        TEST_ASSERT_TRUE(network::publishMqtt(requestTopic, payload.c_str()));
        const uint32_t started = millis();
        while (!received && uint32_t(millis() - started) < 10000) {
            network::mqttLoop();
            delay(10);
        }
        TEST_ASSERT_TRUE_MESSAGE(received, "Split1 server response timeout");
        TEST_ASSERT_NOT_NULL(strstr(response, "\"split\":1"));
        TEST_ASSERT_NOT_NULL(strstr(response, "\"model_version\":\"gesture-cloud-tail-split1-v1.0.0\""));
        String expectedClass = String("\"predicted_class\":\"") + EXPECTED_CLASSES[v] + "\"";
        TEST_ASSERT_NOT_NULL(strstr(response, expectedClass.c_str()));
        const char* confidence = strstr(response, "\"confidence\":");
        TEST_ASSERT_NOT_NULL(confidence);
        const float actualConfidence = atof(confidence + strlen("\"confidence\":"));
        TEST_ASSERT_FLOAT_WITHIN(2e-5f, EXPECTED_CONFIDENCES[v], actualConfidence);
        Serial.printf("SPLIT1_E2E_VECTOR=%u RESPONSE=%s\n", unsigned(v), response);
    }
    network::disconnectMqtt();
    Serial.println("PHASE7_SPLIT1_E2E_PASS");
}
}  // namespace

void setUp() {}
void tearDown() {}
void setup() {
    Serial.begin(115200);
    delay(2000);
    UNITY_BEGIN();
    RUN_TEST(testSplit1EndToEnd);
    UNITY_END();
}
void loop() { delay(1000); }
