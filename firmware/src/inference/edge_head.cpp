#include "edge_head.h"

#include <cfloat>
#include <cmath>

#include "edge_head_data.h"


namespace inference {

bool runEdgeHeadWithMask(
    const float embedding[
        EDGE_HEAD_INPUT_UNITS
    ],
    const uint8_t keepMask[
        EDGE_HEAD_INPUT_UNITS
    ],
    float probabilities[
        EDGE_HEAD_CLASS_COUNT
    ]
) {
    if (
        embedding == nullptr
        || keepMask == nullptr
        || probabilities == nullptr
    ) {
        return false;
    }

    static_assert(
        EDGE_HEAD_INPUT_UNITS
            == edge_head_data::INPUT_UNITS,
        "Edge-head input mismatch."
    );

    static_assert(
        EDGE_HEAD_CLASS_COUNT
            == edge_head_data::CLASS_COUNT,
        "Edge-head class-count mismatch."
    );

    float logits[
        EDGE_HEAD_CLASS_COUNT
    ] = {};

    for (
        size_t outputIndex = 0;
        outputIndex < EDGE_HEAD_CLASS_COUNT;
        ++outputIndex
    ) {
        float sum =
            edge_head_data::BIAS[
                outputIndex
            ];

        for (
            size_t inputIndex = 0;
            inputIndex < EDGE_HEAD_INPUT_UNITS;
            ++inputIndex
        ) {
            const float value =
                embedding[
                    inputIndex
                ];

            if (!std::isfinite(value)) {
                return false;
            }

            const uint8_t keep =
                keepMask[
                    inputIndex
                ];

            if (
                keep != 0
                && keep != 1
            ) {
                return false;
            }

            const float maskedValue =
                keep
                ? (
                    value
                    * edge_head_data::
                        INVERTED_DROPOUT_SCALE
                )
                : 0.0f;

            sum += (
                maskedValue
                * edge_head_data::KERNEL[
                    inputIndex
                ][
                    outputIndex
                ]
            );
        }

        if (!std::isfinite(sum)) {
            return false;
        }

        logits[
            outputIndex
        ] = sum;
    }

    float maxLogit =
        logits[0];

    for (
        size_t classIndex = 1;
        classIndex < EDGE_HEAD_CLASS_COUNT;
        ++classIndex
    ) {
        if (
            logits[classIndex]
            > maxLogit
        ) {
            maxLogit =
                logits[classIndex];
        }
    }

    float denominator = 0.0f;

    for (
        size_t classIndex = 0;
        classIndex < EDGE_HEAD_CLASS_COUNT;
        ++classIndex
    ) {
        const float exponent =
            std::exp(
                logits[classIndex]
                - maxLogit
            );

        if (!std::isfinite(exponent)) {
            return false;
        }

        probabilities[
            classIndex
        ] = exponent;

        denominator += exponent;
    }

    if (
        !std::isfinite(denominator)
        || denominator <= FLT_MIN
    ) {
        return false;
    }

    for (
        size_t classIndex = 0;
        classIndex < EDGE_HEAD_CLASS_COUNT;
        ++classIndex
    ) {
        probabilities[
            classIndex
        ] /= denominator;

        if (
            !std::isfinite(
                probabilities[
                    classIndex
                ]
            )
        ) {
            return false;
        }
    }

    return true;
}


int edgeHeadArgmax(
    const float probabilities[
        EDGE_HEAD_CLASS_COUNT
    ]
) {
    if (probabilities == nullptr) {
        return -1;
    }

    size_t bestIndex = 0;

    for (
        size_t index = 1;
        index < EDGE_HEAD_CLASS_COUNT;
        ++index
    ) {
        if (
            probabilities[index]
            > probabilities[
                bestIndex
            ]
        ) {
            bestIndex = index;
        }
    }

    return static_cast<int>(
        bestIndex
    );
}


}  // namespace inference
