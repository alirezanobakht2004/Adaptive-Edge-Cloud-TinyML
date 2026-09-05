#include <Arduino.h>
#include <unity.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tflm_logits_diagnostic_vectors.h"


namespace {

constexpr size_t TENSOR_ARENA_SIZE =
    32 * 1024;

alignas(16) uint8_t tensorArena[
    TENSOR_ARENA_SIZE
];


bool closeEnough(
    float first,
    float second,
    float tolerance = 1e-6f
) {
    return std::fabs(
        first - second
    ) <= tolerance;
}


int argmax(
    const int8_t values[
        tflm_logits_diagnostic::OUTPUT_COUNT
    ]
) {
    size_t bestIndex = 0;

    for (
        size_t index = 1;
        index
            < tflm_logits_diagnostic::OUTPUT_COUNT;
        ++index
    ) {
        if (
            values[index]
            > values[bestIndex]
        ) {
            bestIndex = index;
        }
    }

    return static_cast<int>(
        bestIndex
    );
}


void testTflmLogitsDiagnostic() {
    const tflite::Model* model =
        tflite::GetModel(
            tflm_logits_diagnostic::MODEL
        );

    TEST_ASSERT_NOT_NULL(
        model
    );

    TEST_ASSERT_EQUAL_INT(
        TFLITE_SCHEMA_VERSION,
        model->version()
    );

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter
        interpreter(
            model,
            resolver,
            tensorArena,
            TENSOR_ARENA_SIZE
        );

    const TfLiteStatus allocationStatus =
        interpreter.AllocateTensors();

    TEST_ASSERT_EQUAL_INT(
        kTfLiteOk,
        allocationStatus
    );

    TfLiteTensor* input =
        interpreter.input(0);

    TfLiteTensor* output =
        interpreter.output(0);

    TEST_ASSERT_NOT_NULL(
        input
    );

    TEST_ASSERT_NOT_NULL(
        output
    );

    TEST_ASSERT_EQUAL_INT(
        kTfLiteInt8,
        input->type
    );

    TEST_ASSERT_EQUAL_INT(
        kTfLiteInt8,
        output->type
    );

    TEST_ASSERT_EQUAL_UINT32(
        tflm_logits_diagnostic::INPUT_COUNT,
        input->bytes
    );

    TEST_ASSERT_EQUAL_UINT32(
        tflm_logits_diagnostic::OUTPUT_COUNT,
        output->bytes
    );

    TEST_ASSERT_TRUE(
        closeEnough(
            input->params.scale,
            tflm_logits_diagnostic::INPUT_SCALE
        )
    );

    TEST_ASSERT_EQUAL_INT(
        tflm_logits_diagnostic::
            INPUT_ZERO_POINT,
        input->params.zero_point
    );

    TEST_ASSERT_TRUE(
        closeEnough(
            output->params.scale,
            tflm_logits_diagnostic::OUTPUT_SCALE
        )
    );

    TEST_ASSERT_EQUAL_INT(
        tflm_logits_diagnostic::
            OUTPUT_ZERO_POINT,
        output->params.zero_point
    );

    int invokeFailures = 0;

    int desktopClassMatches = 0;
    int desktopClassMismatches = 0;

    int esp32Correct = 0;

    int globalMaxLsbDifference = 0;

    uint64_t sumAbsoluteLsbDifference = 0;

    size_t comparedValues = 0;

    size_t globalMaxVectorIndex = 0;
    size_t globalMaxOutputIndex = 0;

    for (
        size_t vectorIndex = 0;
        vectorIndex
            < tflm_logits_diagnostic::VECTOR_COUNT;
        ++vectorIndex
    ) {
        std::memcpy(
            input->data.int8,
            tflm_logits_diagnostic::
                INPUTS[vectorIndex],
            tflm_logits_diagnostic::INPUT_COUNT
        );

        if (
            interpreter.Invoke()
            != kTfLiteOk
        ) {
            ++invokeFailures;

            Serial.printf(
                "INVOKE_FAILURE vector=%u\n",
                static_cast<unsigned>(
                    vectorIndex
                )
            );

            continue;
        }

        int8_t actual[
            tflm_logits_diagnostic::OUTPUT_COUNT
        ];

        std::memcpy(
            actual,
            output->data.int8,
            tflm_logits_diagnostic::OUTPUT_COUNT
        );

        const int desktopClass =
            tflm_logits_diagnostic::
                EXPECTED_CLASSES[
                    vectorIndex
                ];

        const int trueClass =
            tflm_logits_diagnostic::
                TRUE_CLASSES[
                    vectorIndex
                ];

        const int esp32Class =
            argmax(
                actual
            );

        if (
            esp32Class
            == desktopClass
        ) {
            ++desktopClassMatches;
        } else {
            ++desktopClassMismatches;

            Serial.printf(
                "CLASS_MISMATCH "
                "vector=%u "
                "desktop=%d "
                "esp32=%d "
                "true=%d\n",
                static_cast<unsigned>(
                    vectorIndex
                ),
                desktopClass,
                esp32Class,
                trueClass
            );

            Serial.print(
                "  desktop_logits=["
            );

            for (
                size_t outputIndex = 0;
                outputIndex
                    < tflm_logits_diagnostic::
                        OUTPUT_COUNT;
                ++outputIndex
            ) {
                Serial.printf(
                    "%d%s",
                    static_cast<int>(
                        tflm_logits_diagnostic::
                            EXPECTED_LOGITS[
                                vectorIndex
                            ][outputIndex]
                    ),
                    outputIndex + 1
                        == tflm_logits_diagnostic::
                            OUTPUT_COUNT
                        ? ""
                        : ", "
                );
            }

            Serial.println(
                "]"
            );

            Serial.print(
                "  esp32_logits=["
            );

            for (
                size_t outputIndex = 0;
                outputIndex
                    < tflm_logits_diagnostic::
                        OUTPUT_COUNT;
                ++outputIndex
            ) {
                Serial.printf(
                    "%d%s",
                    static_cast<int>(
                        actual[
                            outputIndex
                        ]
                    ),
                    outputIndex + 1
                        == tflm_logits_diagnostic::
                            OUTPUT_COUNT
                        ? ""
                        : ", "
                );
            }

            Serial.println(
                "]"
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
                < tflm_logits_diagnostic::OUTPUT_COUNT;
            ++outputIndex
        ) {
            const int expected =
                static_cast<int>(
                    tflm_logits_diagnostic::
                        EXPECTED_LOGITS[
                            vectorIndex
                        ][outputIndex]
                );

            const int actualValue =
                static_cast<int>(
                    actual[
                        outputIndex
                    ]
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

            ++comparedValues;

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
        }

        if (
            vectorIndex == 82
            || vectorIndex == 119
        ) {
            Serial.printf(
                "\nVECTOR %u\n",
                static_cast<unsigned>(
                    vectorIndex
                )
            );

            Serial.print(
                "  desktop_logits=["
            );

            for (
                size_t outputIndex = 0;
                outputIndex
                    < tflm_logits_diagnostic::
                        OUTPUT_COUNT;
                ++outputIndex
            ) {
                Serial.printf(
                    "%d%s",
                    static_cast<int>(
                        tflm_logits_diagnostic::
                            EXPECTED_LOGITS[
                                vectorIndex
                            ][outputIndex]
                    ),
                    outputIndex + 1
                        == tflm_logits_diagnostic::
                            OUTPUT_COUNT
                        ? ""
                        : ", "
                );
            }

            Serial.println(
                "]"
            );

            Serial.print(
                "  esp32_logits=["
            );

            for (
                size_t outputIndex = 0;
                outputIndex
                    < tflm_logits_diagnostic::
                        OUTPUT_COUNT;
                ++outputIndex
            ) {
                Serial.printf(
                    "%d%s",
                    static_cast<int>(
                        actual[
                            outputIndex
                        ]
                    ),
                    outputIndex + 1
                        == tflm_logits_diagnostic::
                            OUTPUT_COUNT
                        ? ""
                        : ", "
                );
            }

            Serial.println(
                "]"
            );

            Serial.printf(
                "  desktop_class=%d "
                "esp32_class=%d "
                "true=%d\n",
                desktopClass,
                esp32Class,
                trueClass
            );
        }
    }

    const float meanLsbDifference =
        comparedValues > 0
        ? static_cast<float>(
            sumAbsoluteLsbDifference
        )
            / static_cast<float>(
                comparedValues
            )
        : 0.0f;

    const float maxLogitDifference =
        static_cast<float>(
            globalMaxLsbDifference
        )
        * tflm_logits_diagnostic::
            OUTPUT_SCALE;

    const float meanLogitDifference =
        meanLsbDifference
        * tflm_logits_diagnostic::
            OUTPUT_SCALE;

    const float esp32Accuracy =
        static_cast<float>(
            esp32Correct
        )
        / static_cast<float>(
            tflm_logits_diagnostic::
                VECTOR_COUNT
        );

    Serial.println();

    Serial.println(
        "TFLM LOGITS DIAGNOSTIC SUMMARY"
    );

    Serial.println(
        "------------------------------"
    );

    Serial.printf(
        "Diagnostic model bytes:    %u\n",
        static_cast<unsigned>(
            tflm_logits_diagnostic::
                MODEL_LEN
        )
    );

    Serial.printf(
        "Validation vectors:        %u\n",
        static_cast<unsigned>(
            tflm_logits_diagnostic::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Successful invokes:        %u/%u\n",
        static_cast<unsigned>(
            tflm_logits_diagnostic::
                VECTOR_COUNT
            - invokeFailures
        ),
        static_cast<unsigned>(
            tflm_logits_diagnostic::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Desktop class matches:     %d/%u\n",
        desktopClassMatches,
        static_cast<unsigned>(
            tflm_logits_diagnostic::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Desktop class mismatches:  %d\n",
        desktopClassMismatches
    );

    Serial.printf(
        "ESP32 correct:             %d/%u\n",
        esp32Correct,
        static_cast<unsigned>(
            tflm_logits_diagnostic::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "ESP32 validation accuracy: %.6f\n",
        esp32Accuracy
    );

    Serial.printf(
        "Compared logits:           %u\n",
        static_cast<unsigned>(
            comparedValues
        )
    );

    Serial.printf(
        "Max INT8 logits LSB diff:  %d\n",
        globalMaxLsbDifference
    );

    Serial.printf(
        "Mean INT8 logits LSB diff: %.6f\n",
        meanLsbDifference
    );

    Serial.printf(
        "Max dequantized logit diff: %.8f\n",
        maxLogitDifference
    );

    Serial.printf(
        "Mean dequantized logit diff: %.8f\n",
        meanLogitDifference
    );

    Serial.printf(
        "Max diff location:         "
        "vector_%u_output_%u\n",
        static_cast<unsigned>(
            globalMaxVectorIndex
        ),
        static_cast<unsigned>(
            globalMaxOutputIndex
        )
    );

    TEST_ASSERT_EQUAL_INT(
        0,
        invokeFailures
    );

    // This is the diagnostic hard gate.
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
        testTflmLogitsDiagnostic
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}