#include "local_model.h"

#include <Chirale_TensorFlowLite.h>

#include <cmath>
#include <cstring>

#include "deployment_preprocessing_params.h"
#include "gesture_model_data.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


namespace inference {

namespace {

constexpr size_t TENSOR_ARENA_SIZE =
    32 * 1024;

alignas(16) uint8_t tensorArena[
    TENSOR_ARENA_SIZE
];

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter =
    nullptr;

TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

bool initialized = false;


bool closeEnough(
    float first,
    float second,
    float tolerance = 1e-6f
) {
    return std::fabs(
        first - second
    ) <= tolerance;
}

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

    static tflite::AllOpsResolver resolver;

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
        != kTfLiteInt8
    ) {
        return false;
    }

    if (
        outputTensor->type
        != kTfLiteInt8
    ) {
        return false;
    }

    if (
        inputTensor->bytes
        != MODEL_INPUT_FEATURES
    ) {
        return false;
    }

    if (
        outputTensor->bytes
        != LOCAL_CLASS_COUNT
    ) {
        return false;
    }

    if (
        !closeEnough(
            inputTensor->params.scale,
            deployment_preprocessing::
                INPUT_SCALE
        )
    ) {
        return false;
    }

    if (
        inputTensor->params.zero_point
        != deployment_preprocessing::
            INPUT_ZERO_POINT
    ) {
        return false;
    }

    if (
        !closeEnough(
            outputTensor->params.scale,
            deployment_preprocessing::
                OUTPUT_SCALE
        )
    ) {
        return false;
    }

    if (
        outputTensor->params.zero_point
        != deployment_preprocessing::
            OUTPUT_ZERO_POINT
    ) {
        return false;
    }

    initialized = true;

    return true;
}


bool runLocalModel(
    const int8_t input[MODEL_INPUT_FEATURES],
    int8_t output[LOCAL_CLASS_COUNT]
) {
    if (
        !initialized
        || input == nullptr
        || output == nullptr
    ) {
        return false;
    }

    std::memcpy(
        inputTensor->data.int8,
        input,
        MODEL_INPUT_FEATURES
    );

    if (
        interpreter->Invoke()
        != kTfLiteOk
    ) {
        return false;
    }

    std::memcpy(
        output,
        outputTensor->data.int8,
        LOCAL_CLASS_COUNT
    );

    return true;
}


int localModelArgmax(
    const int8_t output[LOCAL_CLASS_COUNT]
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


float dequantizeLocalOutput(
    int8_t value
) {
    return (
        static_cast<float>(value)
        - static_cast<float>(
            deployment_preprocessing::
                OUTPUT_ZERO_POINT
        )
    ) * deployment_preprocessing::
        OUTPUT_SCALE;
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

}  // namespace inference