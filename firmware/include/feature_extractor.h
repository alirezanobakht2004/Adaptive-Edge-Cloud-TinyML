#pragma once

#include <stddef.h>

namespace features {

constexpr size_t WINDOW_SAMPLES = 100;
constexpr size_t SENSOR_CHANNELS = 6;
constexpr size_t FEATURE_COUNT = 10;

bool extractFeaturesV1(
    const float window[WINDOW_SAMPLES][SENSOR_CHANNELS],
    float output[FEATURE_COUNT]
);

}  // namespace features