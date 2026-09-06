#include <Arduino.h>
#include <unity.h>

#include "network/wifi_manager.h"


namespace {

void testProductionWifiManager() {
    Serial.println();
    Serial.println(
        "PHASE6_WIFI_MANAGER_START"
    );

    const bool connected =
        network::connectWifi();

    Serial.printf(
        "WIFI_MANAGER_CONNECT=%s\n",
        connected ? "PASS" : "FAIL"
    );

    TEST_ASSERT_TRUE_MESSAGE(
        connected,
        "production Wi-Fi manager failed to connect"
    );

    TEST_ASSERT_TRUE(
        network::isWifiConnected()
    );

    Serial.printf(
        "WIFI_MANAGER_RSSI_DBM=%ld\n",
        static_cast<long>(
            network::wifiRssiDbm()
        )
    );

    Serial.println(
        "PHASE6_WIFI_MANAGER_PASS"
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
        testProductionWifiManager
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
