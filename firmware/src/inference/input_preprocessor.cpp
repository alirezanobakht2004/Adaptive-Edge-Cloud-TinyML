#include "input_preprocessor.h"

#include <cmath>
#include <stdint.h>

#include "deployment_preprocessing_params.h"


namespace inference {

bool normalizeFeaturesV1(
    const float features[MODEL_INPUT_FEATURES],
    float normalized[MODEL_INPUT_FEATURES]
) {
    if (features == nullptr || normalized == nullptr) {
        return false;
    }

    static_assert(
        MODEL_INPUT_FEATURES
            == deployment_preprocessing::FEATURE_COUNT,
        "Deployment feature count mismatch."
    );

    for (
        size_t index = 0;
        index < MODEL_INPUT_FEATURES;
        ++index
    ) {
        const float value = features[index];

        const float mean =
            deployment_preprocessing::
                NORMALIZATION_MEAN[index];

        const float stdValue =
            deployment_preprocessing::
                NORMALIZATION_STD[index];

        if (
            !std::isfinite(value)
            || !std::isfinite(mean)
            || !std::isfinite(stdValue)
            || stdValue <= 0.0f
        ) {
            return false;
        }

        normalized[index] =
            (value - mean) / stdValue;

        if (!std::isfinite(normalized[index])) {
            return false;
        }
    }

    return true;
}


bool quantizeNormalizedFeatures(
    const float normalized[MODEL_INPUT_FEATURES],
    int8_t quantized[MODEL_INPUT_FEATURES]
) {
    if (normalized == nullptr || quantized == nullptr) {
        return false;
    }

    constexpr float scale =
        deployment_preprocessing::INPUT_SCALE;

    constexpr int32_t zeroPoint =
        deployment_preprocessing::INPUT_ZERO_POINT;

    static_assert(
        scale > 0.0f,
        "INT8 input scale must be positive."
    );

    for (
        size_t index = 0;
        index < MODEL_INPUT_FEATURES;
        ++index
    ) {
        const float value = normalized[index];

        if (!std::isfinite(value)) {
            return false;
        }

        const float transformed =
            value / scale
            + static_cast<float>(zeroPoint);

        long rounded = std::lrint(
            transformed
        );

        if (rounded < -128L) {
            rounded = -128L;
        } else if (rounded > 127L) {
            rounded = 127L;
        }

        quantized[index] =
            static_cast<int8_t>(rounded);
    }

    return true;
}


bool preprocessFeaturesV1(
    const float features[MODEL_INPUT_FEATURES],
    float normalized[MODEL_INPUT_FEATURES],
    int8_t quantized[MODEL_INPUT_FEATURES]
) {
    if (
        !normalizeFeaturesV1(
            features,
            normalized
        )
    ) {
        return false;
    }

    return quantizeNormalizedFeatures(
        normalized,
        quantized
    );
}

}  // namespace inference