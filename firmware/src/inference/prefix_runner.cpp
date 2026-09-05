#include "prefix_runner.h"

#include <Chirale_TensorFlowLite.h>

#include <cmath>
#include <cstring>

#include "prefix_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


namespace inference {

namespace {

// Same conservative known-good arena capacity used for the Phase-4
// Float32 runtime. This is intentionally not claimed as the minimum.
constexpr size_t TENSOR_ARENA_SIZE =
    64 * 1024;

alignas(16) uint8_t tensorArena[
    TENSOR_ARENA_SIZE
];

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter =
    nullptr;

TfLiteTensor* inputTensor = nullptr;

TfLiteTensor* outputTensor = nullptr;

bool initialized = false;

}  // namespace


bool initPrefixRunner() {
    if (initialized) {
        return true;
    }

    static_assert(
        PREFIX_INPUT_FEATURES
            == prefix_model_data::INPUT_FEATURES,
        "Prefix input-size mismatch."
    );

    static_assert(
        PREFIX_OUTPUT_UNITS
            == prefix_model_data::OUTPUT_UNITS,
        "Prefix output-size mismatch."
    );

    model = tflite::GetModel(
        prefix_model_data::MODEL
    );

    if (model == nullptr) {
        return false;
    }

    if (
        model->version()
        != TFLITE_SCHEMA_VERSION
    ) {
        return false;
    }

    static tflite::MicroMutableOpResolver<1>
        resolver;

    static bool resolverAttempted = false;
    static bool resolverReady = false;

    if (!resolverAttempted) {
        resolverAttempted = true;

        resolverReady =
            resolver.AddFullyConnected()
                == kTfLiteOk;
    }

    if (!resolverReady) {
        return false;
    }

    static tflite::MicroInterpreter
        staticInterpreter(
            model,
            resolver,
            tensorArena,
            TENSOR_ARENA_SIZE
        );

    interpreter = &staticInterpreter;

    if (
        interpreter->AllocateTensors()
        != kTfLiteOk
    ) {
        return false;
    }

    inputTensor = interpreter->input(0);
    outputTensor = interpreter->output(0);

    if (
        inputTensor == nullptr
        || outputTensor == nullptr
    ) {
        return false;
    }

    if (
        inputTensor->type
            != kTfLiteFloat32
        || outputTensor->type
            != kTfLiteFloat32
    ) {
        return false;
    }

    if (
        inputTensor->bytes
        != PREFIX_INPUT_FEATURES
            * sizeof(float)
    ) {
        return false;
    }

    if (
        outputTensor->bytes
        != PREFIX_OUTPUT_UNITS
            * sizeof(float)
    ) {
        return false;
    }

    initialized = true;

    return true;
}


bool runPrefixB3(
    const float normalizedInput[
        PREFIX_INPUT_FEATURES
    ],
    float embedding[
        PREFIX_OUTPUT_UNITS
    ]
) {
    if (
        !initialized
        || normalizedInput == nullptr
        || embedding == nullptr
    ) {
        return false;
    }

    for (
        size_t index = 0;
        index < PREFIX_INPUT_FEATURES;
        ++index
    ) {
        if (
            !std::isfinite(
                normalizedInput[index]
            )
        ) {
            return false;
        }
    }

    std::memcpy(
        inputTensor->data.f,
        normalizedInput,
        PREFIX_INPUT_FEATURES
            * sizeof(float)
    );

    if (
        interpreter->Invoke()
        != kTfLiteOk
    ) {
        return false;
    }

    for (
        size_t index = 0;
        index < PREFIX_OUTPUT_UNITS;
        ++index
    ) {
        const float value =
            outputTensor->data.f[index];

        if (!std::isfinite(value)) {
            return false;
        }

        embedding[index] = value;
    }

    return true;
}


size_t prefixRunnerTensorArenaCapacityBytes() {
    return TENSOR_ARENA_SIZE;
}


size_t prefixRunnerTensorArenaUsedBytes() {
    if (
        !initialized
        || interpreter == nullptr
    ) {
        return 0;
    }

    return interpreter->arena_used_bytes();
}


}  // namespace inference
