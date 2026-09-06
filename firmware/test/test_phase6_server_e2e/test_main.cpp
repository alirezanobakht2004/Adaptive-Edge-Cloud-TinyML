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
    "esp32-01-phase6-e2e";

constexpr const char* REQUEST_TOPIC =
    "gesture/esp32-01/inference/request";

constexpr const char* RESPONSE_TOPIC =
    "gesture/esp32-01/inference/response";

constexpr const char* REQUEST_ID =
    "m7-esp32-e2e-001";

constexpr const char* REQUEST_PAYLOAD =
    "{"
    "\"request_id\":\"m7-esp32-e2e-001\","
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

constexpr uint32_t RESPONSE_TIMEOUT_MS =
    10000;


bool responseReceived = false;
bool responseTooLarge = false;

char receivedTopic[128] = {};
char receivedPayload[512] = {};


void onMqttMessage(
    const char* topic,
    const uint8_t* payload,
    unsigned int length
) {
    if (
        topic == nullptr
        || payload == nullptr
    ) {
        return;
    }

    if (
        std::strcmp(
            topic,
            RESPONSE_TOPIC
        ) != 0
    ) {
        return;
    }

    const size_t topicLength =
        std::strlen(topic);

    if (
        topicLength >= sizeof(receivedTopic)
        || length >= sizeof(receivedPayload)
    ) {
        responseTooLarge = true;
        responseReceived = true;
        return;
    }

    std::memcpy(
        receivedTopic,
        topic,
        topicLength
    );

    receivedTopic[topicLength] =
        '\0';

    std::memcpy(
        receivedPayload,
        payload,
        length
    );

    receivedPayload[length] =
        '\0';

    responseReceived = true;
}


void testEsp32ServerEndToEnd() {
    Serial.println();
    Serial.println(
        "PHASE6_SERVER_E2E_START"
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

    const bool subscribed =
        network::subscribeMqtt(
            RESPONSE_TOPIC
        );

    Serial.printf(
        "E2E_SUBSCRIBE=%s\n",
        subscribed ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        subscribed,
        "response subscription failed"
    );


    const bool published =
        network::publishMqtt(
            REQUEST_TOPIC,
            REQUEST_PAYLOAD
        );

    Serial.printf(
        "E2E_REQUEST_PUBLISH=%s\n",
        published ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        published,
        "inference request publish failed"
    );


    const uint32_t startMs =
        millis();

    while (
        !responseReceived
        && static_cast<uint32_t>(
            millis() - startMs
        ) < RESPONSE_TIMEOUT_MS
    ) {
        network::mqttLoop();
        delay(10);
    }


    Serial.printf(
        "E2E_RESPONSE_RECEIVE=%s\n",
        responseReceived
            ? "PASS"
            : "TIMEOUT"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        responseReceived,
        "server inference response timeout"
    );

    TEST_ASSERT_FALSE_MESSAGE(
        responseTooLarge,
        "server response exceeded test buffer"
    );


    Serial.print(
        "E2E_RESPONSE_TOPIC="
    );
    Serial.println(
        receivedTopic
    );

    Serial.print(
        "E2E_RESPONSE_PAYLOAD="
    );
    Serial.println(
        receivedPayload
    );


    TEST_ASSERT_EQUAL_STRING(
        RESPONSE_TOPIC,
        receivedTopic
    );

    TEST_ASSERT_NOT_NULL(
        std::strstr(
            receivedPayload,
            "\"request_id\":\"m7-esp32-e2e-001\""
        )
    );

    TEST_ASSERT_NOT_NULL(
        std::strstr(
            receivedPayload,
            "\"predicted_class\":"
        )
    );

    TEST_ASSERT_NOT_NULL(
        std::strstr(
            receivedPayload,
            "\"confidence\":"
        )
    );

    TEST_ASSERT_NOT_NULL(
        std::strstr(
            receivedPayload,
            "\"server_latency_ms\":"
        )
    );

    TEST_ASSERT_NOT_NULL(
        std::strstr(
            receivedPayload,
            "\"model_version\":"
        )
    );


    network::disconnectMqtt();

    Serial.println(
        "PHASE6_SERVER_E2E_PASS"
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
        testEsp32ServerEndToEnd
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
