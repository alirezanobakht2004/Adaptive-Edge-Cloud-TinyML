#include <Arduino.h>
#include <unity.h>
#include "prefix_runner.h"
#include "uncertainty.h"
#include "policy/adaptive_runtime.h"
#include "network/wifi_manager.h"
#include "network/mqtt_client.h"
#include "failover_config.h"
#include "../test_phase9_meta_policy_parity/meta_vectors.h"

unsigned localCalls = 0, testedWindows = 0;
policy::DecisionResult runWindow(unsigned vector, const char* id) {
    const unsigned before = localCalls;
    float embedding[32]; inference::UncertaintyResult cached{};
    inference::seedUncertaintyMaskPrng(meta_vectors::SEEDS[vector]);
    ++localCalls;
    TEST_ASSERT_TRUE(inference::runPrefixB3(meta_vectors::FEATURES[vector], embedding));
    TEST_ASSERT_TRUE(inference::runStochasticUncertaintyFromEmbedding(embedding, cached));
    TEST_ASSERT_EQUAL(meta_vectors::LOCAL_PREDICTIONS[vector], cached.predictedClass);
    policy::DecisionResult result;
    TEST_ASSERT_TRUE(policy::processCachedDecision(meta_vectors::FEATURES[vector], cached,
        meta_vectors::RAW[vector], id, ++testedWindows, result, true, localCalls-before));
    TEST_ASSERT_EQUAL(1, localCalls-before);
    TEST_ASSERT_EQUAL(1, result.localInferenceCount);
    TEST_ASSERT_EQUAL(0, result.secondInferenceCount);
    TEST_ASSERT_EQUAL(meta_vectors::ACTIONS[vector], result.policy.action);
    if (result.effectiveAction == 0) TEST_ASSERT_EQUAL(cached.predictedClass, result.finalPrediction);
    return result;
}
void testLocalWithoutNetwork() {
    auto r=runWindow(0,"m10-local-normal");
    TEST_ASSERT_FALSE(r.failover); TEST_ASSERT_EQUAL(0,r.txBytes);
}
void testCloudSuccess() {
    TEST_ASSERT_TRUE(policy::connectAdaptiveNetwork());
    auto r=runWindow(9,"m10-cloud-success");
    TEST_ASSERT_FALSE(r.failover); TEST_ASSERT_EQUAL(1,r.effectiveAction);
}
void testWifiUnavailable() {
    network::disconnectMqtt(); network::disconnectWifi(); delay(100);
    TEST_ASSERT_FALSE(network::isWifiConnected());
    auto r=runWindow(9,"m10-wifi-unavailable");
    TEST_ASSERT_TRUE(r.failover);
    TEST_ASSERT_EQUAL(int(policy::FailoverReason::WIFI_UNAVAILABLE),int(r.failoverReason));
    TEST_ASSERT_FALSE(r.requestIssued);
}
void testMqttUnavailable() {
    TEST_ASSERT_TRUE(policy::connectAdaptiveNetwork()); network::disconnectMqtt();
    TEST_ASSERT_TRUE(network::isWifiConnected()); TEST_ASSERT_FALSE(network::isMqttConnected());
    auto r=runWindow(9,"m10-mqtt-unavailable");
    TEST_ASSERT_TRUE(r.failover);
    TEST_ASSERT_EQUAL(int(policy::FailoverReason::MQTT_UNAVAILABLE),int(r.failoverReason));
}
void testServerTimeoutAndNextWindow() {
    TEST_ASSERT_TRUE(policy::connectAdaptiveNetwork());
    Serial.println("M10_SERVER_STOP_REQUEST"); delay(2000);
    auto r=runWindow(9,"m10-server-timeout");
    TEST_ASSERT_TRUE(r.wifiConnected); TEST_ASSERT_TRUE(r.mqttConnected);
    TEST_ASSERT_TRUE(r.requestIssued); TEST_ASSERT_TRUE(r.failover);
    TEST_ASSERT_EQUAL(int(policy::FailoverReason::CLOUD_RESPONSE_TIMEOUT),int(r.failoverReason));
    TEST_ASSERT_GREATER_OR_EQUAL(failover_config::CLOUD_RESPONSE_TIMEOUT_MS*1000, r.roundTripUs);
    auto next=runWindow(0,"m10-next-local");
    TEST_ASSERT_TRUE(next.success); TEST_ASSERT_FALSE(next.failover);
}
void testRecovery() {
    Serial.println("M10_SERVER_START_REQUEST"); delay(2000);
    auto r=runWindow(9,"m10-cloud-recovered");
    TEST_ASSERT_TRUE(r.success); TEST_ASSERT_FALSE(r.failover); TEST_ASSERT_EQUAL(1,r.effectiveAction);
}
void setUp() {}
void tearDown() {}
void setup() {
    Serial.begin(115200); delay(2000); UNITY_BEGIN();
    TEST_ASSERT_TRUE(inference::initPrefixRunner()); TEST_ASSERT_TRUE(policy::initializeMeta());
    RUN_TEST(testLocalWithoutNetwork); RUN_TEST(testCloudSuccess); RUN_TEST(testWifiUnavailable);
    RUN_TEST(testMqttUnavailable); RUN_TEST(testServerTimeoutAndNextWindow); RUN_TEST(testRecovery);
    Serial.printf("M10_FAILOVER_WINDOWS=%u LOCAL_CALLS=%u\n",testedWindows,localCalls);
    UNITY_END();
}
void loop(){delay(1000);}
