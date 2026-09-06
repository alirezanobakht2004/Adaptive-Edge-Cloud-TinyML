#pragma once

#include <Arduino.h>


namespace network {

constexpr uint32_t DEFAULT_WIFI_TIMEOUT_MS =
    20000;


bool connectWifi(
    uint32_t timeoutMs =
        DEFAULT_WIFI_TIMEOUT_MS
);

bool isWifiConnected();

int32_t wifiRssiDbm();

void disconnectWifi();

}  // namespace network
