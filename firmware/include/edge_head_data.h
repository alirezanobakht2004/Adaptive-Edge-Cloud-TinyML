#pragma once

#include <stddef.h>

namespace edge_head_data {

constexpr size_t INPUT_UNITS = 32;
constexpr size_t CLASS_COUNT = 5;
constexpr float DROPOUT_RATE = 0.2f;
constexpr float KEEP_PROBABILITY = 0.8f;
constexpr float INVERTED_DROPOUT_SCALE = 1.25f;

extern const float KERNEL[INPUT_UNITS][CLASS_COUNT];
extern const float BIAS[CLASS_COUNT];

}  // namespace edge_head_data
