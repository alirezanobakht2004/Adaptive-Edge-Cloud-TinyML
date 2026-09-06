#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include <unity.h>

#include "secrets.h"


namespace {

constexpr const char* DEVICE_ID =
    "esp32-01";

constexpr const char* BROKER_HOST =
    "192.168.137.1";

constexpr uint16_t BROKER_PORT =
    1883;

constexpr const char* STATUS_TOPIC =
    "gesture/esp32-01/status";

constexpr const char* STATUS_PAYLOAD =
    "{\"device_id\":\"esp32-01\","
    "\"status\":\"phase6_mqtt_publish_pass\"}";

constexpr uint32_t WIFI_TIMEOUT_MS =
    20000;


WiFiClient networkClient;
PubSubClient mqttClient(networkClient);


bool connectWifi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(
        WIFI_SSID,
        WIFI_PASSWORD
    );

    const uint32_t startMs = millis();

    while (
        WiFi.status() != WL_CONNECTED
        && millis() - startMs < WIFI_TIMEOUT_MS
    ) {
        delay(250);
    }

    return WiFi.status() == WL_CONNECTED;
}


void testMqttConnectAndPublish() {
    Serial.println();
    Serial.println(
        "PHASE6_MQTT_PUBLISH_START"
    );

    const bool wifiConnected =
        connectWifi();

    Serial.printf(
        "WIFI_CONNECT=%s\n",
        wifiConnected ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        wifiConnected,
        "ESP32 Wi-Fi connection failed"
    );

    Serial.print("WIFI_IP=");
    Serial.println(WiFi.localIP());

    Serial.printf(
        "WIFI_RSSI_DBM=%d\n",
        WiFi.RSSI()
    );


    mqttClient.setServer(
        BROKER_HOST,
        BROKER_PORT
    );

    const bool mqttConnected =
        mqttClient.connect(
            "esp32-01-phase6-mqtt-test"
        );

    Serial.printf(
        "MQTT_CONNECT=%s state=%d\n",
        mqttConnected ? "PASS" : "FAIL",
        mqttClient.state()
    );

    TEST_ASSERT_TRUE_MESSAGE(
        mqttConnected,
        "ESP32 MQTT CONNECT failed"
    );


    const bool published =
        mqttClient.publish(
            STATUS_TOPIC,
            STATUS_PAYLOAD,
            false
        );

    Serial.printf(
        "MQTT_PUBLISH=%s\n",
        published ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        published,
        "ESP32 MQTT publish failed"
    );

    mqttClient.loop();
    delay(250);

    mqttClient.disconnect();

    Serial.println(
        "PHASE6_MQTT_PUBLISH_PASS"
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
        testMqttConnectAndPublish
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
