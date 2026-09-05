#include "uncertainty.h"

#include <cmath>


namespace inference {

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
            mean[classIndex]
            > mean[bestIndex]
        ) {
            bestIndex = classIndex;
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
