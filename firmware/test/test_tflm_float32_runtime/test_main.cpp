#include <Arduino.h>
#include <unity.h>

#include <cmath>
#include <cstring>
#include <stdint.h>

#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tflm_float32_runtime_vectors.h"


namespace {

// Diagnostic configuration only.
// This is not a measured deployment memory result.
constexpr size_t TENSOR_ARENA_SIZE =
    64 * 1024;

alignas(16) uint8_t tensorArena[
    TENSOR_ARENA_SIZE
];


int argmax(
    const float values[
        tflm_float32_runtime::OUTPUT_COUNT
    ]
) {
    size_t bestIndex = 0;

    for (
        size_t index = 1;
        index
            < tflm_float32_runtime::OUTPUT_COUNT;
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


void testFloat32RuntimeParity() {
    const tflite::Model* model =
        tflite::GetModel(
            tflm_float32_runtime::MODEL
        );

    TEST_ASSERT_NOT_NULL(
        model
    );

    TEST_ASSERT_EQUAL_INT(
        TFLITE_SCHEMA_VERSION,
        model->version()
    );

    static tflite::MicroMutableOpResolver<2>
        resolver;

    TEST_ASSERT_EQUAL_INT(
        kTfLiteOk,
        resolver.AddFullyConnected()
    );

    TEST_ASSERT_EQUAL_INT(
        kTfLiteOk,
        resolver.AddSoftmax()
    );

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

    TEST_ASSERT_NOT_NULL(
        input
    );

    TEST_ASSERT_NOT_NULL(
        output
    );

    TEST_ASSERT_EQUAL_INT(
        kTfLiteFloat32,
        input->type
    );

    TEST_ASSERT_EQUAL_INT(
        kTfLiteFloat32,
        output->type
    );

    TEST_ASSERT_EQUAL_UINT32(
        tflm_float32_runtime::INPUT_COUNT
            * sizeof(float),
        input->bytes
    );

    TEST_ASSERT_EQUAL_UINT32(
        tflm_float32_runtime::OUTPUT_COUNT
            * sizeof(float),
        output->bytes
    );

    int invokeFailures = 0;
    int classMismatches = 0;
    int esp32Correct = 0;

    float maxAbsDifference = 0.0f;

    double sumAbsDifference = 0.0;

    size_t comparedValues = 0;

    size_t maxVectorIndex = 0;
    size_t maxOutputIndex = 0;

    for (
        size_t vectorIndex = 0;
        vectorIndex
            < tflm_float32_runtime::VECTOR_COUNT;
        ++vectorIndex
    ) {
        std::memcpy(
            input->data.f,
            tflm_float32_runtime::
                INPUTS[vectorIndex],
            tflm_float32_runtime::
                INPUT_COUNT
                * sizeof(float)
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

        const int esp32Class =
            argmax(
                output->data.f
            );

        const int desktopClass =
            tflm_float32_runtime::
                EXPECTED_CLASSES[
                    vectorIndex
                ];

        const int trueClass =
            tflm_float32_runtime::
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
                < tflm_float32_runtime::
                    OUTPUT_COUNT;
            ++outputIndex
        ) {
            const float expected =
                tflm_float32_runtime::
                    EXPECTED_OUTPUTS[
                        vectorIndex
                    ][outputIndex];

            const float actual =
                output->data.f[
                    outputIndex
                ];

            const float difference =
                std::fabs(
                    actual
                    - expected
                );

            sumAbsDifference +=
                static_cast<double>(
                    difference
                );

            ++comparedValues;

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
        }
    }

    const float meanAbsDifference =
        comparedValues > 0
        ? static_cast<float>(
            sumAbsDifference
            / static_cast<double>(
                comparedValues
            )
        )
        : 0.0f;

    const float accuracy =
        static_cast<float>(
            esp32Correct
        )
        / static_cast<float>(
            tflm_float32_runtime::
                VECTOR_COUNT
        );

    Serial.println();

    Serial.println(
        "FLOAT32 TFLM RUNTIME SUMMARY"
    );

    Serial.println(
        "----------------------------"
    );

    Serial.printf(
        "Model bytes:               %u\n",
        static_cast<unsigned>(
            tflm_float32_runtime::
                MODEL_LEN
        )
    );

    Serial.printf(
        "Validation vectors:        %u\n",
        static_cast<unsigned>(
            tflm_float32_runtime::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Successful invokes:        %u/%u\n",
        static_cast<unsigned>(
            tflm_float32_runtime::
                VECTOR_COUNT
            - invokeFailures
        ),
        static_cast<unsigned>(
            tflm_float32_runtime::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Desktop class matches:     %u/%u\n",
        static_cast<unsigned>(
            tflm_float32_runtime::
                VECTOR_COUNT
            - classMismatches
        ),
        static_cast<unsigned>(
            tflm_float32_runtime::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Desktop class mismatches:  %d\n",
        classMismatches
    );

    Serial.printf(
        "ESP32 correct:             %d/%u\n",
        esp32Correct,
        static_cast<unsigned>(
            tflm_float32_runtime::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "ESP32 validation accuracy: %.6f\n",
        accuracy
    );

    Serial.printf(
        "Compared outputs:          %u\n",
        static_cast<unsigned>(
            comparedValues
        )
    );

    Serial.printf(
        "Max output abs diff:       %.10f\n",
        maxAbsDifference
    );

    Serial.printf(
        "Mean output abs diff:      %.10f\n",
        meanAbsDifference
    );

    Serial.printf(
        "Max diff location:         "
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

    // Functional runtime parity is the hard gate.
    TEST_ASSERT_EQUAL_INT(
        0,
        classMismatches
    );
}

}  // namespace


void setup() {
    delay(2000);

    Serial.begin(115200);

    delay(500);

    UNITY_BEGIN();

    RUN_TEST(
        testFloat32RuntimeParity
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}