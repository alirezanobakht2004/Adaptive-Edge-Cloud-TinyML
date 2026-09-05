#include "local_model.h"

#include <Chirale_TensorFlowLite.h>

#include <cmath>
#include <cstring>
#include "gesture_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


namespace inference {

namespace {

// Known-good Float32 configuration from the runtime parity diagnostic.
// This is not yet the measured minimum production arena size.
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


bool initLocalModel() {
    if (initialized) {
        return true;
    }

    model = tflite::GetModel(
        gesture_model_data::MODEL
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


    static tflite::MicroMutableOpResolver<2>
        resolver;

    static bool resolverAttempted = false;
    static bool resolverReady = false;

    if (!resolverAttempted) {
        resolverAttempted = true;

        resolverReady =
            resolver.AddFullyConnected()
                == kTfLiteOk
            && resolver.AddSoftmax()
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
    ) {
        return false;
    }

    if (
        outputTensor->type
        != kTfLiteFloat32
    ) {
        return false;
    }


    if (
        inputTensor->bytes
        != MODEL_INPUT_FEATURES
            * sizeof(float)
    ) {
        return false;
    }

    if (
        outputTensor->bytes
        != LOCAL_CLASS_COUNT
            * sizeof(float)
    ) {
        return false;
    }


    initialized = true;

    return true;
}


bool runLocalModel(
    const float input[MODEL_INPUT_FEATURES],
    float output[LOCAL_CLASS_COUNT]
) {
    if (
        !initialized
        || input == nullptr
        || output == nullptr
    ) {
        return false;
    }


    for (
        size_t index = 0;
        index < MODEL_INPUT_FEATURES;
        ++index
    ) {
        if (!std::isfinite(input[index])) {
            return false;
        }
    }


    std::memcpy(
        inputTensor->data.f,
        input,
        MODEL_INPUT_FEATURES
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
        index < LOCAL_CLASS_COUNT;
        ++index
    ) {
        if (
            !std::isfinite(
                outputTensor->data.f[index]
            )
        ) {
            return false;
        }
    }


    std::memcpy(
        output,
        outputTensor->data.f,
        LOCAL_CLASS_COUNT
            * sizeof(float)
    );

    return true;
}


int localModelArgmax(
    const float output[LOCAL_CLASS_COUNT]
) {
    if (output == nullptr) {
        return -1;
    }

    size_t bestIndex = 0;

    for (
        size_t index = 1;
        index < LOCAL_CLASS_COUNT;
        ++index
    ) {
        if (
            output[index]
            > output[bestIndex]
        ) {
            bestIndex = index;
        }
    }

    return static_cast<int>(
        bestIndex
    );
}


const char* localClassName(
    size_t classIndex
) {
    static const char* const names[
        LOCAL_CLASS_COUNT
    ] = {
        "IDLE",
        "SWIPE_LEFT",
        "SWIPE_RIGHT",
        "ROTATE_CW",
        "SHAKE",
    };

    if (classIndex >= LOCAL_CLASS_COUNT) {
        return "UNKNOWN";
    }

    return names[classIndex];
}


size_t localModelTensorArenaCapacityBytes() {
    return TENSOR_ARENA_SIZE;
}


size_t localModelTensorArenaUsedBytes() {
    if (
        !initialized
        || interpreter == nullptr
    ) {
        return 0;
    }

    return interpreter->arena_used_bytes();
}


}  // namespace inference
