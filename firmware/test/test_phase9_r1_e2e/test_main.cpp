#include <Arduino.h>
#include <unity.h>
#include "prefix_runner.h"
#include "uncertainty.h"
#include "policy/adaptive_runtime.h"
#include "../test_phase9_meta_policy_parity/meta_vectors.h"

void testCachedProductionPaths() {
    TEST_ASSERT_TRUE(inference::initPrefixRunner());
    TEST_ASSERT_TRUE(policy::initializeMeta());
    TEST_ASSERT_TRUE(policy::connectAdaptiveNetwork());
    const unsigned indices[2] = {0, 9};
    for (unsigned v : indices) {
        float embedding[32];
        inference::UncertaintyResult cached{};
        inference::seedUncertaintyMaskPrng(meta_vectors::SEEDS[v]);
        TEST_ASSERT_TRUE(inference::runPrefixB3(meta_vectors::FEATURES[v], embedding));
        TEST_ASSERT_TRUE(inference::runStochasticUncertaintyFromEmbedding(embedding, cached));
        TEST_ASSERT_EQUAL(meta_vectors::LOCAL_PREDICTIONS[v], cached.predictedClass);
        policy::DecisionResult result;
        const char* id = v == 0 ? "r1-controlled-local" : "r1-controlled-cloud";
        TEST_ASSERT_TRUE(policy::processCachedDecision(meta_vectors::FEATURES[v], cached,
            meta_vectors::RAW[v], id, v, result, true));
        TEST_ASSERT_EQUAL(meta_vectors::ACTIONS[v], result.policy.action);
        TEST_ASSERT_EQUAL(0, result.secondInferenceCount);
        if (v == 0) {
            TEST_ASSERT_EQUAL(cached.predictedClass, result.finalPrediction);
            TEST_ASSERT_EQUAL(0, result.txBytes);
            TEST_ASSERT_EQUAL(0, result.rxBytes);
        } else {
            TEST_ASSERT_GREATER_THAN(0, result.txBytes);
            TEST_ASSERT_GREATER_THAN(0, result.rxBytes);
            TEST_ASSERT_EQUAL(2, result.finalPrediction); // Oracle audit: session_01 window 481 (SWIPE_RIGHT).
        }
    }
    Serial.println("R1_PRODUCTION_CORE_E2E_PASS controlled=true features=10 second_inference_count=0");
}
void setUp() {}
void tearDown() {}
void setup() { Serial.begin(115200); delay(2000); UNITY_BEGIN(); RUN_TEST(testCachedProductionPaths); UNITY_END(); }
void loop() { delay(1000); }
