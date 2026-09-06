#include <Arduino.h>
#include <cstring>
#include <unity.h>

#include "network/mqtt_client.h"
#include "network/wifi_manager.h"


namespace {

constexpr const char* BROKER_HOST =
    "192.168.137.1";

constexpr uint16_t BROKER_PORT =
    1883;

constexpr const char* MQTT_CLIENT_ID =
    "esp32-01-phase6-timeout";

constexpr const char* REQUEST_TOPIC =
    "gesture/esp32-01/inference/request";

constexpr const char* RESPONSE_TOPIC =
    "gesture/esp32-01/inference/response";

constexpr const char* REQUEST_ID =
    "m7-timeout-001";

constexpr const char* REQUEST_PAYLOAD =
    "{"
    "\"request_id\":\"m7-timeout-001\","
    "\"device_id\":\"esp32-01\","
    "\"timestamp_ms\":123456,"
    "\"split\":3,"
    "\"embedding\":["
    "0,0,0,0,0,0,0,0,"
    "0,0,0,0,0,0,0,0,"
    "0,0,0,0,0,0,0,0,"
    "0,0,0,0,0,0,0,0"
    "],"
    "\"model_version\":\"gesture-model-v1.1.0\""
    "}";

/*
 * Test-only timeout.
 *
 * This is NOT yet the final production/adaptive threshold.
 * Its purpose is only to prove bounded waiting in Phase 6 / M7.
 */
constexpr uint32_t RESPONSE_TIMEOUT_MS =
    2000;


bool matchingResponseReceived = false;
bool responseTooLarge = false;

char receivedPayload[512] = {};


void onMqttMessage(
    const char* topic,
    const uint8_t* payload,
    unsigned int length
) {
    if (
        topic == nullptr
        || payload == nullptr
        || std::strcmp(
            topic,
            RESPONSE_TOPIC
        ) != 0
    ) {
        return;
    }

    if (length >= sizeof(receivedPayload)) {
        responseTooLarge = true;
        return;
    }

    std::memcpy(
        receivedPayload,
        payload,
        length
    );

    receivedPayload[length] =
        '\0';

    if (
        std::strstr(
            receivedPayload,
            "\"request_id\":\"m7-timeout-001\""
        ) != nullptr
    ) {
        matchingResponseReceived = true;
    }
}


void testServerResponseTimeout() {
    Serial.println();
    Serial.println(
        "PHASE6_SERVER_TIMEOUT_START"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::connectWifi(),
        "Wi-Fi connection failed"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::configureMqtt(
            BROKER_HOST,
            BROKER_PORT
        ),
        "MQTT configuration failed"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::setMqttMessageHandler(
            onMqttMessage
        ),
        "MQTT callback configuration failed"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::connectMqtt(
            MQTT_CLIENT_ID
        ),
        "MQTT connection failed"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::subscribeMqtt(
            RESPONSE_TOPIC
        ),
        "response subscription failed"
    );

    const bool published =
        network::publishMqtt(
            REQUEST_TOPIC,
            REQUEST_PAYLOAD
        );

    Serial.printf(
        "TIMEOUT_REQUEST_PUBLISH=%s\n",
        published ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        published,
        "timeout-test request publish failed"
    );


    const uint32_t startMs =
        millis();

    while (
        !matchingResponseReceived
        && static_cast<uint32_t>(
            millis() - startMs
        ) < RESPONSE_TIMEOUT_MS
    ) {
        network::mqttLoop();
        delay(10);
    }

    const uint32_t elapsedMs =
        static_cast<uint32_t>(
            millis() - startMs
        );


    Serial.printf(
        "SERVER_RESPONSE_RECEIVED=%s\n",
        matchingResponseReceived
            ? "YES"
            : "NO"
    );

    Serial.printf(
        "SERVER_TIMEOUT_DETECTED=%s\n",
        !matchingResponseReceived
            ? "PASS"
            : "FAIL"
    );

    Serial.printf(
        "TIMEOUT_ELAPSED_MS=%lu\n",
        static_cast<unsigned long>(
            elapsedMs
        )
    );


    TEST_ASSERT_FALSE_MESSAGE(
        responseTooLarge,
        "unexpected oversized MQTT response"
    );

    TEST_ASSERT_FALSE_MESSAGE(
        matchingResponseReceived,
        "server response received; stop server before running timeout test"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        elapsedMs >= RESPONSE_TIMEOUT_MS,
        "wait ended before timeout interval"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::isMqttConnected(),
        "MQTT broker connection was lost during server timeout"
    );


    network::disconnectMqtt();

    Serial.println(
        "PHASE6_SERVER_TIMEOUT_PASS"
    );
}

}  // namespace


void setUp() {
}


void tearDown() {
}


void setup() {
    Serial.begin(115200);
    delay(2000);

    UNITY_BEGIN();

    RUN_TEST(
        testServerResponseTimeout
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
