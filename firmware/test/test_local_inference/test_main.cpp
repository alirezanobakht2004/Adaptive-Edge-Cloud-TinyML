#include <Arduino.h>
#include <unity.h>

#include <cmath>
#include <cstdlib>
#include <stdint.h>

#include "deployment_preprocessing_params.h"
#include "local_model.h"
#include "local_inference_vectors.h"


namespace {

void testFullValidationLocalInferenceParity() {
    TEST_ASSERT_TRUE(
        inference::initLocalModel()
    );

    int invokeFailures = 0;

    int desktopClassMatches = 0;
    int desktopClassMismatches = 0;

    int esp32Correct = 0;

    int globalMaxLsbDifference = 0;

    float globalMaxProbabilityDifference =
        0.0f;

    uint64_t sumAbsoluteLsbDifference = 0;

    size_t comparedOutputs = 0;

    size_t globalMaxVectorIndex = 0;
    size_t globalMaxOutputIndex = 0;

    for (
        size_t vectorIndex = 0;
        vectorIndex
            < local_inference_vectors::VECTOR_COUNT;
        ++vectorIndex
    ) {
        int8_t actual[
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

        if (esp32Class == desktopClass) {
            ++desktopClassMatches;
        } else {
            ++desktopClassMismatches;

            Serial.printf(
                "CLASS_MISMATCH "
                "vector=%u "
                "label=%s "
                "desktop=%d "
                "esp32=%d "
                "true=%d\n",
                static_cast<unsigned>(
                    vectorIndex
                ),
                local_inference_vectors::
                    LABELS[vectorIndex],
                desktopClass,
                esp32Class,
                trueClass
            );
        }

        if (esp32Class == trueClass) {
            ++esp32Correct;
        }

        for (
            size_t outputIndex = 0;
            outputIndex
                < inference::LOCAL_CLASS_COUNT;
            ++outputIndex
        ) {
            const int expected =
                static_cast<int>(
                    local_inference_vectors::
                        EXPECTED_OUTPUTS[
                            vectorIndex
                        ][outputIndex]
                );

            const int actualValue =
                static_cast<int>(
                    actual[outputIndex]
                );

            const int difference =
                std::abs(
                    actualValue
                    - expected
                );

            sumAbsoluteLsbDifference +=
                static_cast<uint64_t>(
                    difference
                );

            ++comparedOutputs;

            const float probabilityDifference =
                static_cast<float>(
                    difference
                )
                * deployment_preprocessing::
                    OUTPUT_SCALE;

            if (
                difference
                > globalMaxLsbDifference
            ) {
                globalMaxLsbDifference =
                    difference;

                globalMaxVectorIndex =
                    vectorIndex;

                globalMaxOutputIndex =
                    outputIndex;
            }

            if (
                probabilityDifference
                > globalMaxProbabilityDifference
            ) {
                globalMaxProbabilityDifference =
                    probabilityDifference;
            }
        }
    }

    const float meanAbsoluteLsbDifference =
        comparedOutputs > 0
        ? static_cast<float>(
            sumAbsoluteLsbDifference
        )
            / static_cast<float>(
                comparedOutputs
            )
        : 0.0f;

    const float meanAbsoluteProbabilityDifference =
        meanAbsoluteLsbDifference
        * deployment_preprocessing::
            OUTPUT_SCALE;

    const float esp32Accuracy =
        static_cast<float>(
            esp32Correct
        )
        / static_cast<float>(
            local_inference_vectors::VECTOR_COUNT
        );

    Serial.println();
    Serial.println(
        "FULL VALIDATION LOCAL INFERENCE"
    );
    Serial.println(
        "-------------------------------"
    );

    Serial.printf(
        "Validation vectors:       %u\n",
        static_cast<unsigned>(
            local_inference_vectors::VECTOR_COUNT
        )
    );

    Serial.printf(
        "Successful invokes:       %u/%u\n",
        static_cast<unsigned>(
            local_inference_vectors::VECTOR_COUNT
            - invokeFailures
        ),
        static_cast<unsigned>(
            local_inference_vectors::VECTOR_COUNT
        )
    );

    Serial.printf(
        "Desktop class matches:    %d/%u\n",
        desktopClassMatches,
        static_cast<unsigned>(
            local_inference_vectors::VECTOR_COUNT
        )
    );

    Serial.printf(
        "Desktop class mismatches: %d\n",
        desktopClassMismatches
    );

    Serial.printf(
        "ESP32 correct:            %d/%u\n",
        esp32Correct,
        static_cast<unsigned>(
            local_inference_vectors::VECTOR_COUNT
        )
    );

    Serial.printf(
        "ESP32 validation accuracy: %.6f\n",
        esp32Accuracy
    );

    Serial.printf(
        "Compared output values:   %u\n",
        static_cast<unsigned>(
            comparedOutputs
        )
    );

    Serial.printf(
        "Max INT8 LSB diff:        %d\n",
        globalMaxLsbDifference
    );

    Serial.printf(
        "Mean INT8 LSB diff:       %.6f\n",
        meanAbsoluteLsbDifference
    );

    Serial.printf(
        "Max probability diff:     %.8f\n",
        globalMaxProbabilityDifference
    );

    Serial.printf(
        "Mean probability diff:    %.8f\n",
        meanAbsoluteProbabilityDifference
    );

    Serial.printf(
        "Max diff location:        "
        "vector_%u_output_%u\n",
        static_cast<unsigned>(
            globalMaxVectorIndex
        ),
        static_cast<unsigned>(
            globalMaxOutputIndex
        )
    );

    // Deployment must invoke successfully for every
    // validation vector.
    TEST_ASSERT_EQUAL_INT(
        0,
        invokeFailures
    );

    // Functional parity is the hard gate here.
    TEST_ASSERT_EQUAL_INT(
        0,
        desktopClassMismatches
    );
}

}  // namespace


void setup() {
    delay(2000);

    Serial.begin(115200);

    delay(500);

    UNITY_BEGIN();

    RUN_TEST(
        testFullValidationLocalInferenceParity
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}