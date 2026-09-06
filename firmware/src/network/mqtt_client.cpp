#include "mqtt_client.h"

#include <PubSubClient.h>
#include <WiFi.h>

#include "wifi_manager.h"


namespace {

WiFiClient transportClient;
PubSubClient client(transportClient);

bool configured = false;

network::MqttMessageHandler messageHandler =
    nullptr;


void dispatchMqttMessage(
    char* topic,
    uint8_t* payload,
    unsigned int length
) {
    if (messageHandler == nullptr) {
        return;
    }

    messageHandler(
        topic,
        payload,
        length
    );
}

}  // namespace


namespace network {

bool configureMqtt(
    const char* brokerHost,
    uint16_t brokerPort
) {
    if (
        brokerHost == nullptr
        || brokerHost[0] == '\0'
        || brokerPort == 0
    ) {
        configured = false;
        return false;
    }

    client.setServer(
        brokerHost,
        brokerPort
    );

    client.setCallback(
        dispatchMqttMessage
    );

    if (
        !client.setBufferSize(
            DEFAULT_MQTT_BUFFER_BYTES
        )
    ) {
        configured = false;
        return false;
    }

    configured = true;
    return true;
}


bool setMqttMessageHandler(
    MqttMessageHandler handler
) {
    messageHandler = handler;

    return messageHandler
        != nullptr;
}


bool connectMqtt(
    const char* clientId
) {
    if (
        !configured
        || clientId == nullptr
        || clientId[0] == '\0'
        || !isWifiConnected()
    ) {
        return false;
    }

    if (client.connected()) {
        return true;
    }

    return client.connect(
        clientId
    );
}


bool isMqttConnected() {
    return client.connected();
}


bool subscribeMqtt(
    const char* topic
) {
    if (
        !client.connected()
        || topic == nullptr
        || topic[0] == '\0'
    ) {
        return false;
    }

    return client.subscribe(
        topic
    );
}


bool publishMqtt(
    const char* topic,
    const char* payload,
    bool retained
) {
    if (
        !client.connected()
        || topic == nullptr
        || topic[0] == '\0'
        || payload == nullptr
    ) {
        return false;
    }

    return client.publish(
        topic,
        payload,
        retained
    );
}


bool mqttLoop() {
    if (!client.connected()) {
        return false;
    }

    return client.loop();
}


int mqttState() {
    return client.state();
}


void disconnectMqtt() {
    if (client.connected()) {
        client.disconnect();
    }
}

}  // namespace network
