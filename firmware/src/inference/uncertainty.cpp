#include "uncertainty.h"

#include <cmath>
#include <cstring>

#include <esp_system.h>

#include "edge_head_data.h"


namespace inference {

namespace {

// xorshift32 is deliberately used as a tiny, deterministic,
// non-cryptographic PRNG for MC-Dropout masks.
//
// esp_random() is used only to seed this PRNG in the production path.
// We do not claim that esp_random() provides continuous true entropy when
// Wi-Fi/Bluetooth or another documented entropy source is not enabled.
constexpr uint32_t FALLBACK_PRNG_SEED =
    0xA341316Cu;

// floor(0.8 * 2^32) rounded to the nearest usable strict threshold.
// A uint32 sample is kept when sample < KEEP_THRESHOLD.
constexpr uint32_t KEEP_THRESHOLD =
    3435973837u;

static_assert(
    edge_head_data::DROPOUT_RATE
        == 0.2f,
    "Runtime mask generator expects Dropout(0.2)."
);

static_assert(
    edge_head_data::KEEP_PROBABILITY
        == 0.8f,
    "Runtime mask generator expects keep probability 0.8."
);

uint32_t prngState =
    FALLBACK_PRNG_SEED;


uint32_t nextXorshift32() {
    uint32_t value =
        prngState;

    value ^= (
        value << 13
    );

    value ^= (
        value >> 17
    );

    value ^= (
        value << 5
    );

    if (value == 0u) {
        value =
            FALLBACK_PRNG_SEED;
    }

    prngState =
        value;

    return value;
}


uint8_t countUniqueMasks(
    const uint8_t masks[
        UNCERTAINTY_PASS_COUNT
    ][
        EDGE_HEAD_INPUT_UNITS
    ]
) {
    uint8_t uniqueCount = 0;

    for (
        size_t passIndex = 0;
        passIndex < UNCERTAINTY_PASS_COUNT;
        ++passIndex
    ) {
        bool seenBefore = false;

        for (
            size_t previousIndex = 0;
            previousIndex < passIndex;
            ++previousIndex
        ) {
            if (
                std::memcmp(
                    masks[
                        passIndex
                    ],
                    masks[
                        previousIndex
                    ],
                    EDGE_HEAD_INPUT_UNITS
                        * sizeof(uint8_t)
                )
                == 0
            ) {
                seenBefore = true;
                break;
            }
        }

        if (!seenBefore) {
            ++uniqueCount;
        }
    }

    return uniqueCount;
}


float computeMaxPassProbabilityRange(
    const float passProbabilities[
        UNCERTAINTY_PASS_COUNT
    ][
        EDGE_HEAD_CLASS_COUNT
    ]
) {
    float maxRange = 0.0f;

    for (
        size_t classIndex = 0;
        classIndex < EDGE_HEAD_CLASS_COUNT;
        ++classIndex
    ) {
        float minimum =
            passProbabilities[
                0
            ][
                classIndex
            ];

        float maximum =
            minimum;

        for (
            size_t passIndex = 1;
            passIndex < UNCERTAINTY_PASS_COUNT;
            ++passIndex
        ) {
            const float value =
                passProbabilities[
                    passIndex
                ][
                    classIndex
                ];

            if (value < minimum) {
                minimum = value;
            }

            if (value > maximum) {
                maximum = value;
            }
        }

        const float range =
            maximum - minimum;

        if (range > maxRange) {
            maxRange = range;
        }
    }

    return maxRange;
}

}  // namespace


void seedUncertaintyMaskPrng(
    uint32_t seed
) {
    prngState =
        (
            seed == 0u
        )
        ? FALLBACK_PRNG_SEED
        : seed;
}


uint32_t seedUncertaintyMaskPrngFromDevice() {
    uint32_t seed =
        esp_random();

    // Mix a second device RNG-state read so an accidental zero word does
    // not become the only source value. This is a seed, not a claim of
    // cryptographic entropy.
    seed ^= (
        esp_random()
        * 0x9E3779B9u
    );

    if (seed == 0u) {
        seed =
            FALLBACK_PRNG_SEED;
    }

    seedUncertaintyMaskPrng(
        seed
    );

    return seed;
}


bool generateUncertaintyKeepMask(
    uint8_t keepMask[
        EDGE_HEAD_INPUT_UNITS
    ],
    uint8_t* keptUnitCount
) {
    if (keepMask == nullptr) {
        return false;
    }

    uint8_t kept = 0;

    for (
        size_t unitIndex = 0;
        unitIndex < EDGE_HEAD_INPUT_UNITS;
        ++unitIndex
    ) {
        const uint8_t keep =
            (
                nextXorshift32()
                < KEEP_THRESHOLD
            )
            ? 1u
            : 0u;

        keepMask[
            unitIndex
        ] = keep;

        kept = static_cast<uint8_t>(
            kept + keep
        );
    }

    if (keptUnitCount != nullptr) {
        *keptUnitCount =
            kept;
    }

    return true;
}


bool runStochasticUncertaintyFromEmbedding(
    const float embedding[
        EDGE_HEAD_INPUT_UNITS
    ],
    UncertaintyResult& result,
    UncertaintyRuntimeDiagnostics* diagnostics
) {
    if (embedding == nullptr) {
        return false;
    }

    uint8_t masks[
        UNCERTAINTY_PASS_COUNT
    ][
        EDGE_HEAD_INPUT_UNITS
    ] = {};

    float passProbabilities[
        UNCERTAINTY_PASS_COUNT
    ][
        EDGE_HEAD_CLASS_COUNT
    ] = {};

    uint8_t keptUnits[
        UNCERTAINTY_PASS_COUNT
    ] = {};

    for (
        size_t passIndex = 0;
        passIndex < UNCERTAINTY_PASS_COUNT;
        ++passIndex
    ) {
        if (
            !generateUncertaintyKeepMask(
                masks[
                    passIndex
                ],
                &keptUnits[
                    passIndex
                ]
            )
        ) {
            return false;
        }

        if (
            !runEdgeHeadWithMask(
                embedding,
                masks[
                    passIndex
                ],
                passProbabilities[
                    passIndex
                ]
            )
        ) {
            return false;
        }
    }

    if (
        !computeUncertaintyFromPasses(
            passProbabilities,
            result
        )
    ) {
        return false;
    }

    if (diagnostics != nullptr) {
        for (
            size_t passIndex = 0;
            passIndex < UNCERTAINTY_PASS_COUNT;
            ++passIndex
        ) {
            diagnostics->keptUnits[
                passIndex
            ] = keptUnits[
                passIndex
            ];
        }

        diagnostics->uniqueMaskCount =
            countUniqueMasks(
                masks
            );

        diagnostics->maxPassProbabilityRange =
            computeMaxPassProbabilityRange(
                passProbabilities
            );
    }

    return true;
}


bool computeUncertaintyFromPasses(
    const float passProbabilities[
        UNCERTAINTY_PASS_COUNT
    ][
        EDGE_HEAD_CLASS_COUNT
    ],
    UncertaintyResult& result
) {
    double mean[
        EDGE_HEAD_CLASS_COUNT
    ] = {};

    for (
        size_t passIndex = 0;
        passIndex < UNCERTAINTY_PASS_COUNT;
        ++passIndex
    ) {
        double rowSum = 0.0;

        for (
            size_t classIndex = 0;
            classIndex < EDGE_HEAD_CLASS_COUNT;
            ++classIndex
        ) {
            const float value =
                passProbabilities[
                    passIndex
                ][
                    classIndex
                ];

            if (
                !std::isfinite(value)
                || value < -1e-5f
                || value > 1.0f + 1e-5f
            ) {
                return false;
            }

            rowSum += value;

            mean[
                classIndex
            ] += value;
        }

        if (
            std::fabs(
                rowSum - 1.0
            )
            > 1e-4
        ) {
            return false;
        }
    }

    for (
        size_t classIndex = 0;
        classIndex < EDGE_HEAD_CLASS_COUNT;
        ++classIndex
    ) {
        mean[
            classIndex
        ] /= static_cast<double>(
            UNCERTAINTY_PASS_COUNT
        );

        result.meanProbabilities[
            classIndex
        ] = static_cast<float>(
            mean[
                classIndex
            ]
        );
    }

    double predictiveEntropy = 0.0;

    for (
        size_t classIndex = 0;
        classIndex < EDGE_HEAD_CLASS_COUNT;
        ++classIndex
    ) {
        const double probability =
            mean[
                classIndex
            ];

        if (probability > 0.0) {
            predictiveEntropy -= (
                probability
                * std::log(
                    probability
                )
            );
        }
    }

    const double maxEntropy =
        std::log(
            static_cast<double>(
                EDGE_HEAD_CLASS_COUNT
            )
        );

    if (
        !std::isfinite(
            predictiveEntropy
        )
        || maxEntropy <= 0.0
    ) {
        return false;
    }

    result.predictiveEntropyNats =
        static_cast<float>(
            predictiveEntropy
        );

    result.normalizedPredictiveEntropy =
        static_cast<float>(
            predictiveEntropy
            / maxEntropy
        );

    double varianceSum = 0.0;
    double maxVariance = 0.0;

    for (
        size_t classIndex = 0;
        classIndex < EDGE_HEAD_CLASS_COUNT;
        ++classIndex
    ) {
        double squaredSum = 0.0;

        for (
            size_t passIndex = 0;
            passIndex < UNCERTAINTY_PASS_COUNT;
            ++passIndex
        ) {
            const double difference =
                static_cast<double>(
                    passProbabilities[
                        passIndex
                    ][
                        classIndex
                    ]
                )
                - mean[
                    classIndex
                ];

            squaredSum += (
                difference
                * difference
            );
        }

        const double variance =
            squaredSum
            / static_cast<double>(
                UNCERTAINTY_PASS_COUNT
            );

        varianceSum += variance;

        if (
            classIndex == 0
            || variance > maxVariance
        ) {
            maxVariance = variance;
        }
    }

    result.meanClassVariance =
        static_cast<float>(
            varianceSum
            / static_cast<double>(
                EDGE_HEAD_CLASS_COUNT
            )
        );

    result.maxClassVariance =
        static_cast<float>(
            maxVariance
        );

    size_t bestIndex = 0;

    for (
        size_t classIndex = 1;
        classIndex < EDGE_HEAD_CLASS_COUNT;
        ++classIndex
    ) {
        if (
            mean[
                classIndex
            ]
            > mean[
                bestIndex
            ]
        ) {
            bestIndex =
                classIndex;
        }
    }

    result.predictedClass =
        static_cast<int>(
            bestIndex
        );

    result.confidence =
        static_cast<float>(
            mean[
                bestIndex
            ]
        );

    if (
        !std::isfinite(
            result.normalizedPredictiveEntropy
        )
        || result.normalizedPredictiveEntropy
            < -1e-6f
        || result.normalizedPredictiveEntropy
            > 1.0f + 1e-6f
        || !std::isfinite(
            result.meanClassVariance
        )
        || !std::isfinite(
            result.maxClassVariance
        )
        || !std::isfinite(
            result.confidence
        )
    ) {
        return false;
    }

    return true;
}


}  // namespace inference
