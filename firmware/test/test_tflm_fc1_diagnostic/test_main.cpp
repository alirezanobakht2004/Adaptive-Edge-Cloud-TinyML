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

#include "tflm_fc1_diagnostic_vectors.h"


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


void printPreview(
    const int8_t* values,
    size_t count
) {
    const size_t previewCount =
        count < 12
        ? count
        : 12;

    Serial.print("[");

    for (
        size_t index = 0;
        index < previewCount;
        ++index
    ) {
        Serial.printf(
            "%d",
            static_cast<int>(
                values[index]
            )
        );

        if (
            index + 1
            < previewCount
        ) {
            Serial.print(", ");
        }
    }

    if (
        count > previewCount
    ) {
        Serial.print(", ...");
    }

    Serial.print("]");
}


void testTflmFc1Diagnostic() {
    const tflite::Model* model =
        tflite::GetModel(
            tflm_fc1_diagnostic::MODEL
        );

    TEST_ASSERT_NOT_NULL(model);

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

    TEST_ASSERT_EQUAL_INT(
        kTfLiteOk,
        interpreter.AllocateTensors()
    );

    TfLiteTensor* input =
        interpreter.input(0);

    TfLiteTensor* output =
        interpreter.output(0);

    TEST_ASSERT_NOT_NULL(input);
    TEST_ASSERT_NOT_NULL(output);

    TEST_ASSERT_EQUAL_INT(
        kTfLiteInt8,
        input->type
    );

    TEST_ASSERT_EQUAL_INT(
        kTfLiteInt8,
        output->type
    );

    TEST_ASSERT_EQUAL_UINT32(
        tflm_fc1_diagnostic::
            INPUT_COUNT,
        input->bytes
    );

    TEST_ASSERT_EQUAL_UINT32(
        tflm_fc1_diagnostic::
            OUTPUT_COUNT,
        output->bytes
    );

    TEST_ASSERT_TRUE(
        closeEnough(
            input->params.scale,
            tflm_fc1_diagnostic::
                INPUT_SCALE
        )
    );

    TEST_ASSERT_EQUAL_INT(
        tflm_fc1_diagnostic::
            INPUT_ZERO_POINT,
        input->params.zero_point
    );

    TEST_ASSERT_TRUE(
        closeEnough(
            output->params.scale,
            tflm_fc1_diagnostic::
                OUTPUT_SCALE
        )
    );

    TEST_ASSERT_EQUAL_INT(
        tflm_fc1_diagnostic::
            OUTPUT_ZERO_POINT,
        output->params.zero_point
    );

    int invokeFailures = 0;

    for (
        size_t probeIndex = 0;
        probeIndex
            < tflm_fc1_diagnostic::
                PROBE_COUNT;
        ++probeIndex
    ) {
        std::memcpy(
            input->data.int8,
            tflm_fc1_diagnostic::
                INPUTS[probeIndex],
            tflm_fc1_diagnostic::
                INPUT_COUNT
        );

        if (
            interpreter.Invoke()
            != kTfLiteOk
        ) {
            ++invokeFailures;

            Serial.printf(
                "INVOKE_FAILURE probe=%u\n",
                static_cast<unsigned>(
                    probeIndex
                )
            );

            continue;
        }

        const int8_t* actual =
            output->data.int8;

        const int8_t* expected =
            tflm_fc1_diagnostic::
                EXPECTED_FC1[
                    probeIndex
                ];

        int mismatchCount = 0;
        int maxLsbDifference = 0;

        uint64_t sumLsbDifference = 0;

        size_t maxDifferenceIndex = 0;

        for (
            size_t index = 0;
            index
                < tflm_fc1_diagnostic::
                    OUTPUT_COUNT;
            ++index
        ) {
            const int difference =
                std::abs(
                    static_cast<int>(
                        actual[index]
                    )
                    - static_cast<int>(
                        expected[index]
                    )
                );

            sumLsbDifference +=
                static_cast<uint64_t>(
                    difference
                );

            if (difference != 0) {
                ++mismatchCount;
            }

            if (
                difference
                > maxLsbDifference
            ) {
                maxLsbDifference =
                    difference;

                maxDifferenceIndex =
                    index;
            }
        }

        const float meanLsbDifference =
            static_cast<float>(
                sumLsbDifference
            )
            / static_cast<float>(
                tflm_fc1_diagnostic::
                    OUTPUT_COUNT
            );

        Serial.println();

        Serial.printf(
            "VECTOR %d\n",
            tflm_fc1_diagnostic::
                VALIDATION_INDICES[
                    probeIndex
                ]
        );

        Serial.printf(
            "FC1 mismatches: %d/%u\n",
            mismatchCount,
            static_cast<unsigned>(
                tflm_fc1_diagnostic::
                    OUTPUT_COUNT
            )
        );

        Serial.printf(
            "FC1 max LSB diff: %d\n",
            maxLsbDifference
        );

        Serial.printf(
            "FC1 mean LSB diff: %.6f\n",
            meanLsbDifference
        );

        if (
            maxLsbDifference > 0
        ) {
            Serial.printf(
                "FC1 max diff index: %u\n",
                static_cast<unsigned>(
                    maxDifferenceIndex
                )
            );
        }

        Serial.print(
            "Desktop FC1: "
        );

        printPreview(
            expected,
            tflm_fc1_diagnostic::
                OUTPUT_COUNT
        );

        Serial.println();

        Serial.print(
            "ESP32 FC1:   "
        );

        printPreview(
            actual,
            tflm_fc1_diagnostic::
                OUTPUT_COUNT
        );

        Serial.println();

        Serial.printf(
            "FC1_EXACT_VECTOR_%d=%s\n",
            tflm_fc1_diagnostic::
                VALIDATION_INDICES[
                    probeIndex
                ],
            mismatchCount == 0
                ? "true"
                : "false"
        );
    }

    Serial.println();

    Serial.println(
        "TFLM FC1 DIAGNOSTIC COMPLETE"
    );

    Serial.printf(
        "Invoke failures: %d\n",
        invokeFailures
    );

    // Numeric mismatch is diagnostic evidence,
    // not a harness failure.
    TEST_ASSERT_EQUAL_INT(
        0,
        invokeFailures
    );
}

}  // namespace


void setup() {
    delay(2000);

    Serial.begin(115200);

    delay(500);

    UNITY_BEGIN();

    RUN_TEST(
        testTflmFc1Diagnostic
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}