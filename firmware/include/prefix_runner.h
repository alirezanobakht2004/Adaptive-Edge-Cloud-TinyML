#pragma once

#include <stddef.h>

namespace inference {

constexpr size_t PREFIX_INPUT_FEATURES = 10;
constexpr size_t PREFIX_OUTPUT_UNITS = 32;

bool initPrefixRunner();

bool runPrefixB3(
    const float normalizedInput[
        PREFIX_INPUT_FEATURES
    ],
    float embedding[
        PREFIX_OUTPUT_UNITS
    ]
);

size_t prefixRunnerTensorArenaCapacityBytes();

size_t prefixRunnerTensorArenaUsedBytes();

}  // namespace inference
