#include <Arduino.h>
#include <unity.h>

#include <cmath>

#include "feature_extractor.h"
#include "feature_parity_vectors.h"


namespace {

constexpr float ABSOLUTE_TOLERANCE = 1e-4f;
constexpr float RELATIVE_TOLERANCE = 1e-4f;


void testFeatureParity() {
    float globalMaxAbsDifference = 0.0f;

    for (
        size_t vectorIndex = 0;
        vectorIndex < feature_parity::VECTOR_COUNT;
        ++vectorIndex
    ) {
        float actual[features::FEATURE_COUNT];

        const bool success =
            features::extractFeaturesV1(
                feature_parity::WINDOWS[
                    vectorIndex
                ],
                actual
            );

        TEST_ASSERT_TRUE(success);

        Serial.printf(
            "\nVECTOR %u: %s\n",
            static_cast<unsigned>(vectorIndex),
            feature_parity::LABELS[
                vectorIndex
            ]
        );

        for (
            size_t featureIndex = 0;
            featureIndex < features::FEATURE_COUNT;
            ++featureIndex
        ) {
            const float expected =
                feature_parity::EXPECTED_FEATURES[
                    vectorIndex
                ][featureIndex];

            const float difference =
                std::fabs(
                    actual[featureIndex]
                    - expected
                );

            if (
                difference
                > globalMaxAbsDifference
            ) {
                globalMaxAbsDifference =
                    difference;
            }

            const float tolerance =
                ABSOLUTE_TOLERANCE
                + RELATIVE_TOLERANCE
                * std::fabs(expected);

            Serial.printf(
                "  f%02u "
                "expected=% .8f "
                "actual=% .8f "
                "abs_diff=%.10f "
                "tol=%.10f\n",
                static_cast<unsigned>(
                    featureIndex + 1
                ),
                expected,
                actual[featureIndex],
                difference,
                tolerance
            );

            TEST_ASSERT_FLOAT_WITHIN(
                tolerance,
                expected,
                actual[featureIndex]
            );
        }
    }

    Serial.printf(
        "\nFEATURE_PARITY_MAX_ABS_DIFF="
        "%.10f\n",
        globalMaxAbsDifference
    );
}

}  // namespace


void setup() {
    delay(2000);

    Serial.begin(115200);

    delay(500);

    UNITY_BEGIN();

    RUN_TEST(
        testFeatureParity
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}