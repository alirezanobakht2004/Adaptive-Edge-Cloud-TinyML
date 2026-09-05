#include <Arduino.h>
#include <unity.h>

#include <cstdlib>
#include <cstring>
#include <stdint.h>

#include <Chirale_TensorFlowLite.h>

#include "gesture_model_data.h"
#include "internal_tensor_vectors.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


namespace {

// Diagnostic-only arena.
//
// preserve_all_tensors disables normal activation
// buffer reuse, so this diagnostic intentionally uses
// a larger arena than the normal deployment runner.
//
// This is NOT a measured deployment memory requirement.
constexpr size_t TENSOR_ARENA_SIZE =
    64 * 1024;

alignas(16) uint8_t tensorArena[
    TENSOR_ARENA_SIZE
];


size_t elementCount(
    const TfLiteEvalTensor* tensor
) {
    if (
        tensor == nullptr
        || tensor->dims == nullptr
    ) {
        return 0;
    }

    size_t count = 1;

    for (
        int index = 0;
        index < tensor->dims->size;
        ++index
    ) {
        count *= static_cast<size_t>(
            tensor->dims->data[index]
        );
    }

    return count;
}


void printPreview(
    const int8_t* values,
    size_t count
) {
    const size_t previewCount =
        count < 10
        ? count
        : 10;

    Serial.print(
        "["
    );

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
            Serial.print(
                ", "
            );
        }
    }

    if (
        count > previewCount
    ) {
        Serial.print(
            ", ..."
        );
    }

    Serial.print(
        "]"
    );
}


void testInternalTensorDiagnostic() {
    const tflite::Model* model =
        tflite::GetModel(
            gesture_model_data::MODEL
        );

    TEST_ASSERT_NOT_NULL(
        model
    );

    TEST_ASSERT_EQUAL_INT(
        TFLITE_SCHEMA_VERSION,
        model->version()
    );

    static tflite::AllOpsResolver resolver;

    //
    // The final constructor argument enables
    // preserve_all_tensors.
    //
    static tflite::MicroInterpreter
        interpreter(
            model,
            resolver,
            tensorArena,
            TENSOR_ARENA_SIZE,
            nullptr,
            nullptr,
            true
        );

    TEST_ASSERT_TRUE(
        interpreter.preserve_all_tensors()
    );

    const TfLiteStatus allocationStatus =
        interpreter.AllocateTensors();

    TEST_ASSERT_EQUAL_INT(
        kTfLiteOk,
        allocationStatus
    );

    TfLiteTensor* input =
        interpreter.input(0);

    TEST_ASSERT_NOT_NULL(
        input
    );

    TEST_ASSERT_EQUAL_INT(
        kTfLiteInt8,
        input->type
    );

    TEST_ASSERT_EQUAL_UINT32(
        internal_tensor_vectors::INPUT_COUNT,
        input->bytes
    );

    int invokeFailures = 0;

    for (
        size_t probeIndex = 0;
        probeIndex
            < internal_tensor_vectors::PROBE_COUNT;
        ++probeIndex
    ) {
        const auto& probe =
            internal_tensor_vectors::
                PROBES[probeIndex];

        std::memcpy(
            input->data.int8,
            probe.input,
            internal_tensor_vectors::
                INPUT_COUNT
        );

        if (
            interpreter.Invoke()
            != kTfLiteOk
        ) {
            ++invokeFailures;

            Serial.printf(
                "INVOKE_FAILURE "
                "vector=%d\n",
                probe.validation_index
            );

            continue;
        }

        Serial.println();

        Serial.printf(
            "VECTOR %d true=%d\n",
            probe.validation_index,
            probe.true_class
        );

        const char* firstDivergentStage =
            nullptr;

        int firstDivergentTensor =
            -1;

        for (
            size_t stageIndex = 0;
            stageIndex
                < internal_tensor_vectors::
                    STAGE_COUNT;
            ++stageIndex
        ) {
            const auto& stage =
                probe.stages[
                    stageIndex
                ];

            TfLiteEvalTensor* tensor =
                interpreter.GetTensor(
                    stage.tensor_index
                );

            TEST_ASSERT_NOT_NULL(
                tensor
            );

            TEST_ASSERT_EQUAL_INT(
                kTfLiteInt8,
                tensor->type
            );

            const size_t actualCount =
                elementCount(
                    tensor
                );

            TEST_ASSERT_EQUAL_UINT32(
                stage.value_count,
                actualCount
            );

            const int8_t* actual =
                tensor->data.int8;

            TEST_ASSERT_NOT_NULL(
                actual
            );

            size_t mismatchCount = 0;

            int maxLsbDifference = 0;

            uint64_t sumLsbDifference = 0;

            size_t maxDifferenceIndex = 0;

            for (
                size_t valueIndex = 0;
                valueIndex
                    < stage.value_count;
                ++valueIndex
            ) {
                const int expectedValue =
                    static_cast<int>(
                        stage.expected[
                            valueIndex
                        ]
                    );

                const int actualValue =
                    static_cast<int>(
                        actual[
                            valueIndex
                        ]
                    );

                const int difference =
                    std::abs(
                        actualValue
                        - expectedValue
                    );

                sumLsbDifference +=
                    static_cast<uint64_t>(
                        difference
                    );

                if (
                    difference != 0
                ) {
                    ++mismatchCount;
                }

                if (
                    difference
                    > maxLsbDifference
                ) {
                    maxLsbDifference =
                        difference;

                    maxDifferenceIndex =
                        valueIndex;
                }
            }

            const float meanLsbDifference =
                stage.value_count > 0
                ? static_cast<float>(
                    sumLsbDifference
                )
                    / static_cast<float>(
                        stage.value_count
                    )
                : 0.0f;

            if (
                mismatchCount > 0
                && firstDivergentStage
                    == nullptr
            ) {
                firstDivergentStage =
                    stage.name;

                firstDivergentTensor =
                    stage.tensor_index;
            }

            Serial.printf(
                "  %-7s "
                "tensor=%d "
                "n=%u "
                "mismatches=%u "
                "max_lsb_diff=%d "
                "mean_lsb_diff=%.6f",
                stage.name,
                stage.tensor_index,
                static_cast<unsigned>(
                    stage.value_count
                ),
                static_cast<unsigned>(
                    mismatchCount
                ),
                maxLsbDifference,
                meanLsbDifference
            );

            if (
                maxLsbDifference > 0
            ) {
                Serial.printf(
                    " max_diff_index=%u",
                    static_cast<unsigned>(
                        maxDifferenceIndex
                    )
                );
            }

            Serial.println();

            Serial.print(
                "           desktop="
            );

            printPreview(
                stage.expected,
                stage.value_count
            );

            Serial.println();

            Serial.print(
                "           esp32=  "
            );

            printPreview(
                actual,
                stage.value_count
            );

            Serial.println();
        }

        if (
            firstDivergentStage
            == nullptr
        ) {
            Serial.printf(
                "FIRST_DIVERGENT_STAGE_"
                "VECTOR_%d=NONE\n",
                probe.validation_index
            );
        } else {
            Serial.printf(
                "FIRST_DIVERGENT_STAGE_"
                "VECTOR_%d=%s "
                "tensor=%d\n",
                probe.validation_index,
                firstDivergentStage,
                firstDivergentTensor
            );
        }
    }

    Serial.println();

    Serial.println(
        "INTERNAL TENSOR DIAGNOSTIC COMPLETE"
    );

    Serial.printf(
        "Invoke failures: %d\n",
        invokeFailures
    );

    //
    // Numeric divergence is expected to be measured,
    // not treated as a test failure here.
    //
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
        testInternalTensorDiagnostic
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}