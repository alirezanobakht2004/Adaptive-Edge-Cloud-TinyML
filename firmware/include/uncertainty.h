#pragma once

#include <stddef.h>

#include "edge_head.h"

namespace inference {

constexpr size_t UNCERTAINTY_PASS_COUNT = 5;

struct UncertaintyResult {
    float meanProbabilities[
        EDGE_HEAD_CLASS_COUNT
    ];

    float predictiveEntropyNats;
    float normalizedPredictiveEntropy;
    float meanClassVariance;
    float maxClassVariance;
    float confidence;
    int predictedClass;
};

bool computeUncertaintyFromPasses(
    const float passProbabilities[
        UNCERTAINTY_PASS_COUNT
    ][
        EDGE_HEAD_CLASS_COUNT
    ],
    UncertaintyResult& result
);

}  // namespace inference
