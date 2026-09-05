#pragma once

#include <stddef.h>
#include <stdint.h>

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

struct UncertaintyRuntimeDiagnostics {
    uint8_t keptUnits[
        UNCERTAINTY_PASS_COUNT
    ];

    uint8_t uniqueMaskCount;
    float maxPassProbabilityRange;
};

bool computeUncertaintyFromPasses(
    const float passProbabilities[
        UNCERTAINTY_PASS_COUNT
    ][
        EDGE_HEAD_CLASS_COUNT
    ],
    UncertaintyResult& result
);

// Deterministic seed injection is intentionally public so the stochastic
// runtime can be regression-tested without changing its production path.
void seedUncertaintyMaskPrng(
    uint32_t seed
);

// Seed the non-cryptographic MC-Dropout PRNG from ESP32 device RNG state.
// The returned value is the effective non-zero seed.
//
// This does NOT claim cryptographic or "true random" entropy. MC-Dropout
// only needs a varying Bernoulli-mask sequence.
uint32_t seedUncertaintyMaskPrngFromDevice();

bool generateUncertaintyKeepMask(
    uint8_t keepMask[
        EDGE_HEAD_INPUT_UNITS
    ],
    uint8_t* keptUnitCount = nullptr
);

// One logical MC-Dropout inference from an already-computed B3 embedding:
//
// B3 embedding
// -> 5 generated Dropout masks
// -> explicit masked Edge Head
// -> aggregate probabilities
// -> normalized predictive entropy / variance / confidence
//
// No LOCAL/OFFLOAD threshold or policy decision is made here.
bool runStochasticUncertaintyFromEmbedding(
    const float embedding[
        EDGE_HEAD_INPUT_UNITS
    ],
    UncertaintyResult& result,
    UncertaintyRuntimeDiagnostics* diagnostics = nullptr
);

}  // namespace inference
