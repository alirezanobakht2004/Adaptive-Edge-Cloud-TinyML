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
    "esp32-01-phase6-receive";

constexpr const char* RESPONSE_TOPIC =
    "gesture/esp32-01/inference/response";

constexpr const char* EXPECTED_PAYLOAD =
    "{\"transport_test\":"
    "\"phase6_mqtt_receive_pass\"}";

constexpr uint32_t RECEIVE_TIMEOUT_MS =
    5000;


bool messageReceived = false;
bool messageTooLarge = false;

char receivedTopic[128] = {};
char receivedPayload[256] = {};


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

    const size_t topicLength =
        std::strlen(topic);

    if (
        topicLength >= sizeof(receivedTopic)
        || length >= sizeof(receivedPayload)
    ) {
        messageTooLarge = true;
        messageReceived = true;
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

    messageReceived = true;
}


void testProductionMqttReceive() {
    Serial.println();
    Serial.println(
        "PHASE6_MQTT_RECEIVE_START"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::connectWifi(),
        "production Wi-Fi manager failed"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::configureMqtt(
            BROKER_HOST,
            BROKER_PORT
        ),
        "production MQTT configuration failed"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::setMqttMessageHandler(
            onMqttMessage
        ),
        "MQTT message handler configuration failed"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        network::connectMqtt(
            MQTT_CLIENT_ID
        ),
        "production MQTT connect failed"
    );

    const bool subscribed =
        network::subscribeMqtt(
            RESPONSE_TOPIC
        );

    Serial.printf(
        "MQTT_SUBSCRIBE=%s\n",
        subscribed ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        subscribed,
        "MQTT response subscription failed"
    );


    const uint32_t startMs =
        millis();

    while (
        !messageReceived
        && static_cast<uint32_t>(
            millis() - startMs
        ) < RECEIVE_TIMEOUT_MS
    ) {
        network::mqttLoop();
        delay(10);
    }


    Serial.printf(
        "MQTT_RECEIVE=%s\n",
        messageReceived ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        messageReceived,
        "MQTT response was not received"
    );

    TEST_ASSERT_FALSE_MESSAGE(
        messageTooLarge,
        "received MQTT message exceeded test buffers"
    );

    Serial.print(
        "MQTT_RECEIVED_TOPIC="
    );
    Serial.println(
        receivedTopic
    );

    Serial.print(
        "MQTT_RECEIVED_PAYLOAD="
    );
    Serial.println(
        receivedPayload
    );

    TEST_ASSERT_EQUAL_STRING(
        RESPONSE_TOPIC,
        receivedTopic
    );

    TEST_ASSERT_EQUAL_STRING(
        EXPECTED_PAYLOAD,
        receivedPayload
    );

    network::disconnectMqtt();

    Serial.println(
        "PHASE6_MQTT_RECEIVE_PASS"
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
        testProductionMqttReceive
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
