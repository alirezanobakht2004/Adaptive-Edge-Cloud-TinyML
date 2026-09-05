#pragma once

#include <stddef.h>
#include <stdint.h>

namespace inference {

constexpr size_t MODEL_INPUT_FEATURES = 10;

bool normalizeFeaturesV1(
    const float features[MODEL_INPUT_FEATURES],
    float normalized[MODEL_INPUT_FEATURES]
);

bool quantizeNormalizedFeatures(
    const float normalized[MODEL_INPUT_FEATURES],
    int8_t quantized[MODEL_INPUT_FEATURES]
);

bool preprocessFeaturesV1(
    const float features[MODEL_INPUT_FEATURES],
    float normalized[MODEL_INPUT_FEATURES],
    int8_t quantized[MODEL_INPUT_FEATURES]
);

}  // namespace inference