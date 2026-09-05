#include <Arduino.h>
#include <unity.h>

#include <cmath>
#include <stdint.h>

#include "input_preprocessor.h"
#include "input_preprocessing_vectors.h"


namespace {

constexpr float ABSOLUTE_TOLERANCE = 1e-4f;
constexpr float RELATIVE_TOLERANCE = 1e-4f;


void testInputPreprocessingParity() {
    float maxNormalizedDifference = 0.0f;

    for (
        size_t vectorIndex = 0;
        vectorIndex
            < input_preprocessing_vectors::VECTOR_COUNT;
        ++vectorIndex
    ) {
        float normalized[
            inference::MODEL_INPUT_FEATURES
        ];

        int8_t quantized[
            inference::MODEL_INPUT_FEATURES
        ];

        const bool success =
            inference::preprocessFeaturesV1(
                input_preprocessing_vectors::
                    FEATURES[vectorIndex],
                normalized,
                quantized
            );

        TEST_ASSERT_TRUE(success);

        Serial.printf(
            "\nVECTOR %u: %s\n",
            static_cast<unsigned>(vectorIndex),
            input_preprocessing_vectors::
                LABELS[vectorIndex]
        );

        for (
            size_t featureIndex = 0;
            featureIndex
                < inference::MODEL_INPUT_FEATURES;
            ++featureIndex
        ) {
            const float expected =
                input_preprocessing_vectors::
                    EXPECTED_NORMALIZED[
                        vectorIndex
                    ][featureIndex];

            const float difference =
                std::fabs(
                    normalized[featureIndex]
                    - expected
                );

            if (
                difference
                > maxNormalizedDifference
            ) {
                maxNormalizedDifference =
                    difference;
            }

            const float tolerance =
                ABSOLUTE_TOLERANCE
                + RELATIVE_TOLERANCE
                * std::fabs(expected);

            Serial.printf(
                "  f%02u "
                "norm_expected=% .8f "
                "norm_actual=% .8f "
                "q_expected=%d "
                "q_actual=%d\n",
                static_cast<unsigned>(
                    featureIndex + 1
                ),
                expected,
                normalized[featureIndex],
                static_cast<int>(
                    input_preprocessing_vectors::
                        EXPECTED_INT8[
                            vectorIndex
                        ][featureIndex]
                ),
                static_cast<int>(
                    quantized[featureIndex]
                )
            );

            TEST_ASSERT_FLOAT_WITHIN(
                tolerance,
                expected,
                normalized[featureIndex]
            );

            TEST_ASSERT_EQUAL_INT8(
                input_preprocessing_vectors::
                    EXPECTED_INT8[
                        vectorIndex
                    ][featureIndex],
                quantized[featureIndex]
            );
        }
    }

    Serial.printf(
        "\nPREPROCESSING_MAX_NORMALIZED_ABS_DIFF="
        "%.10f\n",
        maxNormalizedDifference
    );
}

}  // namespace


void setup() {
    delay(2000);

    Serial.begin(115200);

    delay(500);

    UNITY_BEGIN();

    RUN_TEST(
        testInputPreprocessingParity
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}