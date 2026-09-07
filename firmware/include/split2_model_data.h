#pragma once
#include <stddef.h>
namespace split2_model_data {
constexpr size_t SPLIT_ID = 2;
constexpr size_t INPUT_FEATURES = 10;
constexpr size_t OUTPUT_UNITS = 48;
constexpr size_t EXPECTED_MODEL_LEN = 16984;
extern const unsigned char MODEL[];
extern const size_t MODEL_LEN;
extern const char MODEL_SHA256[];
extern const char SOURCE_MODEL_VERSION[];
}
