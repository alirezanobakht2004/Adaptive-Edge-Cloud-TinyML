#pragma once

#include <stddef.h>
#include <stdint.h>

#include "input_preprocessor.h"

namespace inference {

constexpr size_t LOCAL_CLASS_COUNT = 5;

bool initLocalModel();

bool runLocalModel(
    const int8_t input[MODEL_INPUT_FEATURES],
    int8_t output[LOCAL_CLASS_COUNT]
);

int localModelArgmax(
    const int8_t output[LOCAL_CLASS_COUNT]
);

float dequantizeLocalOutput(
    int8_t value
);

const char* localClassName(
    size_t classIndex
);

}  // namespace inference