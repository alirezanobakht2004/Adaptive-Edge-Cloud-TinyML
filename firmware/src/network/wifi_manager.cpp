#include "wifi_manager.h"

#include <WiFi.h>

#include "secrets.h"


namespace network {

bool connectWifi(
    uint32_t timeoutMs
) {
    if (WiFi.status() == WL_CONNECTED) {
        return true;
    }

    WiFi.mode(WIFI_STA);

    WiFi.begin(
        WIFI_SSID,
        WIFI_PASSWORD
    );

    const uint32_t startMs =
        millis();

    while (
        WiFi.status() != WL_CONNECTED
        && static_cast<uint32_t>(
            millis() - startMs
        ) < timeoutMs
    ) {
        delay(250);
    }

    return WiFi.status()
        == WL_CONNECTED;
}


bool isWifiConnected() {
    return WiFi.status()
        == WL_CONNECTED;
}


int32_t wifiRssiDbm() {
    if (!isWifiConnected()) {
        return 0;
    }

    return WiFi.RSSI();
}


void disconnectWifi() {
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
}

}  // namespace network
