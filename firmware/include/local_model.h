#pragma once

#include <stddef.h>

#include "input_preprocessor.h"

namespace inference {

constexpr size_t LOCAL_CLASS_COUNT = 5;

bool initLocalModel();

bool runLocalModel(
    const float input[MODEL_INPUT_FEATURES],
    float output[LOCAL_CLASS_COUNT]
);

int localModelArgmax(
    const float output[LOCAL_CLASS_COUNT]
);

const char* localClassName(
    size_t classIndex
);

}  // namespace inference
