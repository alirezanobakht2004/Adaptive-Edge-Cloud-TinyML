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

// Diagnostic resource introspection.
// The capacity is the configured static arena size.
// The used value becomes valid after initLocalModel().
size_t localModelTensorArenaCapacityBytes();

size_t localModelTensorArenaUsedBytes();

}  // namespace inference
