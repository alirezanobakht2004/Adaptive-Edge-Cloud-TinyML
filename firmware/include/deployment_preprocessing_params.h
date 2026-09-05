#pragma once

#include <stddef.h>
#include <stdint.h>

namespace deployment_preprocessing {

constexpr size_t FEATURE_COUNT = 10;

constexpr char MODEL_VERSION[] = "gesture-model-v1.0.0";
constexpr char INT8_MODEL_SHA256[] = "50221c98a62c546cb37dcf3537433aa2fb9f0050a37aaa833bc4b1c256ac4dd9";

constexpr float NORMALIZATION_MEAN[FEATURE_COUNT] = {
    0.0658589527f, 0.228865713f, 0.00798967574f, 0.0212296583f, 0.0154495975f, 0.0180670395f, -5.23002577f, 7.07336426f, 10.7444363f, 22.1345863f
};

constexpr float NORMALIZATION_STD[FEATURE_COUNT] = {
    0.0836580843f, 0.264868736f, 0.0354624167f, 0.0247841459f, 0.0167471245f, 0.0232806858f, 11.6945019f, 10.9813766f, 15.2749367f, 27.2098846f
};

constexpr float INPUT_SCALE = 0.0500446633f;
constexpr int32_t INPUT_ZERO_POINT = -12;

constexpr float OUTPUT_SCALE = 0.00390625f;
constexpr int32_t OUTPUT_ZERO_POINT = -128;

}  // namespace deployment_preprocessing
