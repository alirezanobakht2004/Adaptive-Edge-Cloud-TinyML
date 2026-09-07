#pragma once

#include <stddef.h>

namespace split1_model_data {

constexpr size_t SPLIT_ID = 1;
constexpr size_t INPUT_FEATURES = 10;
constexpr size_t OUTPUT_UNITS = 64;
constexpr size_t EXPECTED_MODEL_LEN = 3996;

extern const unsigned char MODEL[];
extern const size_t MODEL_LEN;
extern const char SOURCE_MODEL_VERSION[];
extern const char MODEL_SHA256[];

}  // namespace split1_model_data
