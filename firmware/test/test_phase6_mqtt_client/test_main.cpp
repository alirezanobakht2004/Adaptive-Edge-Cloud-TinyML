#include <Arduino.h>
#include <unity.h>

#include "network/mqtt_client.h"
#include "network/wifi_manager.h"


namespace {

constexpr const char* BROKER_HOST =
    "192.168.137.1";

constexpr uint16_t BROKER_PORT =
    1883;

constexpr const char* MQTT_CLIENT_ID =
    "esp32-01-phase6-production";

constexpr const char* STATUS_TOPIC =
    "gesture/esp32-01/status";

constexpr const char* STATUS_PAYLOAD =
    "{\"device_id\":\"esp32-01\","
    "\"status\":\"phase6_mqtt_client_pass\"}";


void testProductionMqttClient() {
    Serial.println();
    Serial.println(
        "PHASE6_MQTT_CLIENT_START"
    );

    const bool wifiConnected =
        network::connectWifi();

    Serial.printf(
        "WIFI_MANAGER_CONNECT=%s\n",
        wifiConnected ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        wifiConnected,
        "production Wi-Fi manager failed"
    );


    const bool configured =
        network::configureMqtt(
            BROKER_HOST,
            BROKER_PORT
        );

    Serial.printf(
        "MQTT_CONFIGURE=%s\n",
        configured ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        configured,
        "production MQTT configuration failed"
    );


    const bool connected =
        network::connectMqtt(
            MQTT_CLIENT_ID
        );

    Serial.printf(
        "MQTT_CLIENT_CONNECT=%s state=%d\n",
        connected ? "PASS" : "FAIL",
        network::mqttState()
    );

    TEST_ASSERT_TRUE_MESSAGE(
        connected,
        "production MQTT client failed to connect"
    );

    TEST_ASSERT_TRUE(
        network::isMqttConnected()
    );


    const bool published =
        network::publishMqtt(
            STATUS_TOPIC,
            STATUS_PAYLOAD
        );

    Serial.printf(
        "MQTT_CLIENT_PUBLISH=%s\n",
        published ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        published,
        "production MQTT publish failed"
    );

    network::mqttLoop();
    delay(250);

    network::disconnectMqtt();

    Serial.println(
        "PHASE6_MQTT_CLIENT_PASS"
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
        testProductionMqttClient
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
