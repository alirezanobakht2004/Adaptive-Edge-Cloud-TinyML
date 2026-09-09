#include <Arduino.h>
#include <unity.h>
#include <cmath>
#include "policy/meta_learner.h"
#include "meta_vectors.h"

void setUp() {}
void tearDown() {}
void testPolicyParity() {
    TEST_ASSERT_TRUE(policy::initializeMeta());
    float maximum = 0;
    unsigned local = 0, cloud = 0;
    uint32_t totalUs = 0;
    for (unsigned i = 0; i < meta_vectors::COUNT; ++i) {
        policy::MetaResult result;
        TEST_ASSERT_TRUE(policy::runMeta(meta_vectors::RAW[i], result));
        for (unsigned j = 0; j < 6; ++j) TEST_ASSERT_FLOAT_WITHIN(1e-5f, meta_vectors::NORMALIZED[i][j], result.normalized[j]);
        for (unsigned j = 0; j < 2; ++j) {
            const float difference = fabsf(result.probabilities[j] - meta_vectors::EXPECTED[i][j]);
            maximum = fmaxf(maximum, difference);
            TEST_ASSERT_FLOAT_WITHIN(1e-5f, meta_vectors::EXPECTED[i][j], result.probabilities[j]);
        }
        TEST_ASSERT_EQUAL_INT(meta_vectors::ACTIONS[i], result.action);
        if (result.action) ++cloud; else ++local;
        totalUs += result.inferenceUs;
    }
    TEST_ASSERT_TRUE(local > 0 && cloud > 0);
    Serial.printf("R1_POLICY_PARITY vectors=%u local=%u cloud=%u max_abs_diff=%.9g total_invoke_us=%u arena_used=%u heap=%u\n",
                  meta_vectors::COUNT, local, cloud, double(maximum), totalUs, unsigned(policy::metaArenaUsed()), ESP.getFreeHeap());
    Serial.println("R1_POLICY_PARITY_PASS");
}
void setup() { Serial.begin(115200); delay(2000); UNITY_BEGIN(); RUN_TEST(testPolicyParity); UNITY_END(); }
void loop() { delay(1000); }
