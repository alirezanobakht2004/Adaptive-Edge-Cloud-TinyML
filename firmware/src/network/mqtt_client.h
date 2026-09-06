#pragma once

#include <Arduino.h>


namespace network {

constexpr uint16_t DEFAULT_MQTT_BUFFER_BYTES =
    1024;


bool configureMqtt(
    const char* brokerHost,
    uint16_t brokerPort
);

bool connectMqtt(
    const char* clientId
);

bool isMqttConnected();

bool publishMqtt(
    const char* topic,
    const char* payload,
    bool retained = false
);

bool mqttLoop();

int mqttState();

void disconnectMqtt();

}  // namespace network
