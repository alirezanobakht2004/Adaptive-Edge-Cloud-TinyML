#include <Arduino.h>
#include <unity.h>

#include <cmath>
#include <cstring>

#include "gesture_model_data.h"
#include "local_model.h"
#include "local_inference_vectors.h"


namespace {

constexpr float OUTPUT_ABS_TOLERANCE =
    1.0e-6f;


void testProductionFloat32LocalInference() {
    TEST_ASSERT_EQUAL_STRING(
        local_inference_vectors::MODEL_SHA256,
        gesture_model_data::MODEL_SHA256
    );

    TEST_ASSERT_EQUAL_UINT32(
        local_inference_vectors::MODEL_LEN,
        gesture_model_data::MODEL_LEN
    );

    TEST_ASSERT_TRUE(
        inference::initLocalModel()
    );

    int invokeFailures = 0;
    int classMismatches = 0;
    int numericViolations = 0;
    int esp32Correct = 0;

    float maxAbsDifference = 0.0f;
    size_t maxVectorIndex = 0;
    size_t maxOutputIndex = 0;

    for (
        size_t vectorIndex = 0;
        vectorIndex
            < local_inference_vectors::VECTOR_COUNT;
        ++vectorIndex
    ) {
        float actual[
            inference::LOCAL_CLASS_COUNT
        ] = {};

        const bool success =
            inference::runLocalModel(
                local_inference_vectors::
                    INPUTS[vectorIndex],
                actual
            );

        if (!success) {
            ++invokeFailures;

            Serial.printf(
                "INVOKE_FAILURE "
                "vector=%u "
                "validation_index=%u "
                "label=%s\n",
                static_cast<unsigned>(
                    vectorIndex
                ),
                static_cast<unsigned>(
                    local_inference_vectors::
                        VALIDATION_INDICES[
                            vectorIndex
                        ]
                ),
                local_inference_vectors::
                    LABELS[vectorIndex]
            );

            continue;
        }

        const int esp32Class =
            inference::localModelArgmax(
                actual
            );

        const int desktopClass =
            local_inference_vectors::
                EXPECTED_CLASSES[
                    vectorIndex
                ];

        const int trueClass =
            local_inference_vectors::
                TRUE_CLASSES[
                    vectorIndex
                ];

        if (
            esp32Class
            != desktopClass
        ) {
            ++classMismatches;

            Serial.printf(
                "CLASS_MISMATCH "
                "vector=%u "
                "validation_index=%u "
                "label=%s "
                "desktop=%d "
                "esp32=%d "
                "true=%d\n",
                static_cast<unsigned>(
                    vectorIndex
                ),
                static_cast<unsigned>(
                    local_inference_vectors::
                        VALIDATION_INDICES[
                            vectorIndex
                        ]
                ),
                local_inference_vectors::
                    LABELS[vectorIndex],
                desktopClass,
                esp32Class,
                trueClass
            );
        }

        if (
            esp32Class
            == trueClass
        ) {
            ++esp32Correct;
        }

        for (
            size_t outputIndex = 0;
            outputIndex
                < inference::LOCAL_CLASS_COUNT;
            ++outputIndex
        ) {
            const float expected =
                local_inference_vectors::
                    EXPECTED_OUTPUTS[
                        vectorIndex
                    ][outputIndex];

            const float difference =
                std::fabs(
                    actual[outputIndex]
                    - expected
                );

            if (
                difference
                > maxAbsDifference
            ) {
                maxAbsDifference =
                    difference;

                maxVectorIndex =
                    vectorIndex;

                maxOutputIndex =
                    outputIndex;
            }

            if (
                difference
                > OUTPUT_ABS_TOLERANCE
            ) {
                ++numericViolations;

                Serial.printf(
                    "NUMERIC_MISMATCH "
                    "vector=%u "
                    "output=%u "
                    "expected=%.10f "
                    "actual=%.10f "
                    "diff=%.10f\n",
                    static_cast<unsigned>(
                        vectorIndex
                    ),
                    static_cast<unsigned>(
                        outputIndex
                    ),
                    expected,
                    actual[outputIndex],
                    difference
                );
            }
        }
    }

    Serial.println();
    Serial.println(
        "PRODUCTION FLOAT32 LOCAL RUNNER"
    );
    Serial.println(
        "-------------------------------"
    );

    Serial.printf(
        "Model bytes:             %u\n",
        static_cast<unsigned>(
            gesture_model_data::MODEL_LEN
        )
    );

    Serial.printf(
        "Vectors:                 %u\n",
        static_cast<unsigned>(
            local_inference_vectors::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Successful invokes:      %u/%u\n",
        static_cast<unsigned>(
            local_inference_vectors::
                VECTOR_COUNT
            - invokeFailures
        ),
        static_cast<unsigned>(
            local_inference_vectors::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Desktop class matches:   %u/%u\n",
        static_cast<unsigned>(
            local_inference_vectors::
                VECTOR_COUNT
            - classMismatches
        ),
        static_cast<unsigned>(
            local_inference_vectors::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "ESP32 correct:           %d/%u\n",
        esp32Correct,
        static_cast<unsigned>(
            local_inference_vectors::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Numeric violations:      %d\n",
        numericViolations
    );

    Serial.printf(
        "Output abs tolerance:    %.10f\n",
        OUTPUT_ABS_TOLERANCE
    );

    Serial.printf(
        "Max output abs diff:     %.10f\n",
        maxAbsDifference
    );

    Serial.printf(
        "Max diff location:       "
        "vector_%u_output_%u\n",
        static_cast<unsigned>(
            maxVectorIndex
        ),
        static_cast<unsigned>(
            maxOutputIndex
        )
    );

    TEST_ASSERT_EQUAL_INT(
        0,
        invokeFailures
    );

    TEST_ASSERT_EQUAL_INT(
        0,
        classMismatches
    );

    TEST_ASSERT_EQUAL_INT(
        0,
        numericViolations
    );

    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(
            local_inference_vectors::
                VECTOR_COUNT
        ),
        esp32Correct
    );
}

}  // namespace


void setup() {
    delay(2000);

    Serial.begin(115200);

    delay(500);

    UNITY_BEGIN();

    RUN_TEST(
        testProductionFloat32LocalInference
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
