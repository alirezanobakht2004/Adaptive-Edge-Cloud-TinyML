#include <Arduino.h>
#include <unity.h>

#include <cmath>
#include <cstring>
#include <stdint.h>

#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "split2_model_data.h"
#include "split2_parity_vectors.h"


namespace {

constexpr size_t TENSOR_ARENA_SIZE =
    64 * 1024;

alignas(16) uint8_t tensorArena[
    TENSOR_ARENA_SIZE
];


void testSplit2Parity() {

    const tflite::Model* model =
        tflite::GetModel(
            split2_model_data::MODEL
        );

    TEST_ASSERT_NOT_NULL(
        model
    );

    TEST_ASSERT_EQUAL_INT(
        TFLITE_SCHEMA_VERSION,
        model->version()
    );


    static tflite::MicroMutableOpResolver<1>
        resolver;

    TEST_ASSERT_EQUAL_INT(
        kTfLiteOk,
        resolver.AddFullyConnected()
    );


    static tflite::MicroInterpreter interpreter(
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
        kTfLiteFloat32,
        input->type
    );

    TEST_ASSERT_EQUAL_INT(
        kTfLiteFloat32,
        output->type
    );


    TEST_ASSERT_EQUAL_UINT32(
        split2_model_data::INPUT_FEATURES
            * sizeof(float),
        input->bytes
    );


    TEST_ASSERT_EQUAL_UINT32(
        split2_model_data::OUTPUT_UNITS
            * sizeof(float),
        output->bytes
    );


    float maxAbsDifference = 0.0f;


    for (
        size_t vectorIndex = 0;
        vectorIndex <
            split2_parity_vectors::VECTOR_COUNT;
        ++vectorIndex
    ) {

        std::memcpy(
            input->data.f,
            split2_parity_vectors::NORMALIZED_INPUTS[
                vectorIndex
            ],
            split2_parity_vectors::INPUT_FEATURES
                * sizeof(float)
        );


        TEST_ASSERT_EQUAL_INT(
            kTfLiteOk,
            interpreter.Invoke()
        );


        for (
            size_t outputIndex = 0;
            outputIndex <
                split2_parity_vectors::OUTPUT_UNITS;
            ++outputIndex
        ) {

            const float actual =
                output->data.f[outputIndex];

            const float expected =
                split2_parity_vectors::EXPECTED_OUTPUTS[
                    vectorIndex
                ][outputIndex];


            TEST_ASSERT_TRUE(
                std::isfinite(actual)
            );


            const float diff =
                std::fabs(
                    actual - expected
                );


            if (
                diff > maxAbsDifference
            ) {
                maxAbsDifference = diff;
            }


            TEST_ASSERT_TRUE_MESSAGE(
                diff <= 1e-5f,
                "Split2 output mismatch"
            );
        }
    }


    Serial.printf(
        "SPLIT2_VECTOR_COUNT=%u\n",
        static_cast<unsigned>(
            split2_parity_vectors::VECTOR_COUNT
        )
    );


    Serial.printf(
        "SPLIT2_MAX_ABS_DIFF=%.9f\n",
        maxAbsDifference
    );


    Serial.println(
        "PHASE7_SPLIT2_PARITY_PASS"
    );
}


}


void setUp() {
}


void tearDown() {
}


void setup() {

    Serial.begin(115200);
    delay(2000);


    UNITY_BEGIN();


    RUN_TEST(
        testSplit2Parity
    );


    UNITY_END();
}


void loop() {
    delay(1000);
}
