#pragma once

#include <stddef.h>

namespace prefix_model_data {

constexpr size_t INPUT_FEATURES = 10;
constexpr size_t OUTPUT_UNITS = 32;

extern const unsigned char MODEL[];
extern const size_t MODEL_LEN;
extern const char MODEL_VERSION[];
extern const char MODEL_SHA256[];

}  // namespace prefix_model_data
