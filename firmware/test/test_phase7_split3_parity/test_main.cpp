#include <Arduino.h>
#include <unity.h>
#include <cmath>
#include "prefix_runner.h"
#include "split3_parity_vectors.h"

void testSplit3Parity() {
    TEST_ASSERT_TRUE(inference::initPrefixRunner());
    float maximum = 0;
    for (size_t v = 0; v < split3_parity_vectors::VECTOR_COUNT; ++v) {
        float embedding[32];
        TEST_ASSERT_TRUE(inference::runPrefixB3(split3_parity_vectors::NORMALIZED_INPUTS[v], embedding));
        for (size_t i = 0; i < 32; ++i) {
            TEST_ASSERT_TRUE(std::isfinite(embedding[i]));
            const float diff = std::fabs(embedding[i] - split3_parity_vectors::EXPECTED_OUTPUTS[v][i]);
            if (diff > maximum) maximum = diff;
            TEST_ASSERT_FLOAT_WITHIN(1e-5f, split3_parity_vectors::EXPECTED_OUTPUTS[v][i], embedding[i]);
        }
    }
    Serial.printf("SPLIT3_VECTOR_COUNT=5\nSPLIT3_MAX_ABS_DIFF=%.9f\n", maximum);
    Serial.println("PHASE7_SPLIT3_PARITY_PASS");
}
void setUp() {}
void tearDown() {}
void setup() {
    Serial.begin(115200);
    delay(2000);
    UNITY_BEGIN();
    RUN_TEST(testSplit3Parity);
    UNITY_END();
}
void loop() { delay(1000); }
