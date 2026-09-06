#include <Arduino.h>
#include <WiFi.h>
#include <unity.h>

#include "secrets.h"


namespace {

constexpr const char* BROKER_HOST = "192.168.137.1";
constexpr uint16_t BROKER_PORT = 1883;

constexpr uint32_t WIFI_TIMEOUT_MS = 20000;


void testWifiAndBrokerTcpReachability() {
    Serial.println();
    Serial.println("PHASE6_NETWORK_REACHABILITY_START");

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

    Serial.printf(
        "WIFI_STATUS=%d\n",
        static_cast<int>(WiFi.status())
    );

    TEST_ASSERT_EQUAL_INT(
        WL_CONNECTED,
        WiFi.status()
    );

    Serial.print("WIFI_IP=");
    Serial.println(WiFi.localIP());

    Serial.printf(
        "WIFI_RSSI_DBM=%d\n",
        WiFi.RSSI()
    );

    WiFiClient client;

    const bool connected = client.connect(
        BROKER_HOST,
        BROKER_PORT
    );

    Serial.printf(
        "BROKER_TCP_CONNECT=%s\n",
        connected ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        connected,
        "ESP32 could not open TCP connection to Mosquitto"
    );

    client.stop();

    Serial.println(
        "PHASE6_NETWORK_REACHABILITY_PASS"
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
        testWifiAndBrokerTcpReachability
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}

