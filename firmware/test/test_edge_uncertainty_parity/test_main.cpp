#include <Arduino.h>

#include <unity.h>

#include <cmath>

#include "edge_head.h"
#include "edge_uncertainty_parity_vectors.h"
#include "prefix_runner.h"
#include "uncertainty.h"


namespace {

constexpr float B3_TOLERANCE =
    1e-4f;

constexpr float PROBABILITY_TOLERANCE =
    1e-4f;

constexpr float UNCERTAINTY_TOLERANCE =
    1e-4f;


float absoluteDifference(
    float left,
    float right
) {
    return std::fabs(
        left - right
    );
}


int argmaxExpected(
    const float values[
        edge_uncertainty_parity::
            CLASS_COUNT
    ]
) {
    size_t bestIndex = 0;

    for (
        size_t index = 1;
        index
            < edge_uncertainty_parity::
                CLASS_COUNT;
        ++index
    ) {
        if (
            values[index]
            > values[bestIndex]
        ) {
            bestIndex = index;
        }
    }

    return static_cast<int>(
        bestIndex
    );
}


void testEsp32FixedMaskParity() {
    TEST_ASSERT_TRUE_MESSAGE(
        inference::initPrefixRunner(),
        "Prefix runner initialization failed."
    );

    float maxB3Diff = 0.0f;
    float maxPassProbabilityDiff = 0.0f;
    float maxMeanProbabilityDiff = 0.0f;
    float maxUncertaintyDiff = 0.0f;
    float maxMeanVarianceDiff = 0.0f;
    float maxMaxVarianceDiff = 0.0f;

    int top1Agreement = 0;

    for (
        size_t vectorIndex = 0;
        vectorIndex
            < edge_uncertainty_parity::
                VECTOR_COUNT;
        ++vectorIndex
    ) {
        float embedding[
            inference::PREFIX_OUTPUT_UNITS
        ] = {};

        TEST_ASSERT_TRUE_MESSAGE(
            inference::runPrefixB3(
                edge_uncertainty_parity::
                    NORMALIZED_INPUTS[
                        vectorIndex
                    ],
                embedding
            ),
            "Prefix invocation failed."
        );

        for (
            size_t unitIndex = 0;
            unitIndex
                < edge_uncertainty_parity::
                    B3_UNITS;
            ++unitIndex
        ) {
            const float difference =
                absoluteDifference(
                    embedding[
                        unitIndex
                    ],
                    edge_uncertainty_parity::
                        EXPECTED_B3[
                            vectorIndex
                        ][
                            unitIndex
                        ]
                );

            if (
                difference
                > maxB3Diff
            ) {
                maxB3Diff =
                    difference;
            }
        }

        float passProbabilities[
            inference::
                UNCERTAINTY_PASS_COUNT
        ][
            inference::
                EDGE_HEAD_CLASS_COUNT
        ] = {};

        for (
            size_t passIndex = 0;
            passIndex
                < edge_uncertainty_parity::
                    PASS_COUNT;
            ++passIndex
        ) {
            TEST_ASSERT_TRUE_MESSAGE(
                inference::runEdgeHeadWithMask(
                    embedding,
                    edge_uncertainty_parity::
                        KEEP_MASKS[
                            vectorIndex
                        ][
                            passIndex
                        ],
                    passProbabilities[
                        passIndex
                    ]
                ),
                "Fixed-mask edge-head invocation failed."
            );

            for (
                size_t classIndex = 0;
                classIndex
                    < edge_uncertainty_parity::
                        CLASS_COUNT;
                ++classIndex
            ) {
                const float difference =
                    absoluteDifference(
                        passProbabilities[
                            passIndex
                        ][
                            classIndex
                        ],
                        edge_uncertainty_parity::
                            EXPECTED_PASS_PROBABILITIES[
                                vectorIndex
                            ][
                                passIndex
                            ][
                                classIndex
                            ]
                    );

                if (
                    difference
                    > maxPassProbabilityDiff
                ) {
                    maxPassProbabilityDiff =
                        difference;
                }
            }
        }

        inference::UncertaintyResult
            result = {};

        TEST_ASSERT_TRUE_MESSAGE(
            inference::
                computeUncertaintyFromPasses(
                    passProbabilities,
                    result
                ),
            "Uncertainty aggregation failed."
        );

        for (
            size_t classIndex = 0;
            classIndex
                < edge_uncertainty_parity::
                    CLASS_COUNT;
            ++classIndex
        ) {
            const float difference =
                absoluteDifference(
                    result.meanProbabilities[
                        classIndex
                    ],
                    edge_uncertainty_parity::
                        EXPECTED_MEAN_PROBABILITIES[
                            vectorIndex
                        ][
                            classIndex
                        ]
                );

            if (
                difference
                > maxMeanProbabilityDiff
            ) {
                maxMeanProbabilityDiff =
                    difference;
            }
        }

        const float uncertaintyDiff =
            absoluteDifference(
                result.
                    normalizedPredictiveEntropy,
                edge_uncertainty_parity::
                    EXPECTED_UNCERTAINTY_SCORE[
                        vectorIndex
                    ]
            );

        if (
            uncertaintyDiff
            > maxUncertaintyDiff
        ) {
            maxUncertaintyDiff =
                uncertaintyDiff;
        }

        const float meanVarianceDiff =
            absoluteDifference(
                result.meanClassVariance,
                edge_uncertainty_parity::
                    EXPECTED_MEAN_CLASS_VARIANCE[
                        vectorIndex
                    ]
            );

        if (
            meanVarianceDiff
            > maxMeanVarianceDiff
        ) {
            maxMeanVarianceDiff =
                meanVarianceDiff;
        }

        const float maxVarianceDiff =
            absoluteDifference(
                result.maxClassVariance,
                edge_uncertainty_parity::
                    EXPECTED_MAX_CLASS_VARIANCE[
                        vectorIndex
                    ]
            );

        if (
            maxVarianceDiff
            > maxMaxVarianceDiff
        ) {
            maxMaxVarianceDiff =
                maxVarianceDiff;
        }

        const int expectedClass =
            argmaxExpected(
                edge_uncertainty_parity::
                    EXPECTED_MEAN_PROBABILITIES[
                        vectorIndex
                    ]
            );

        if (
            result.predictedClass
            == expectedClass
        ) {
            ++top1Agreement;
        }
    }

    Serial.println();
    Serial.println(
        "PHASE 5 / M6 — ESP32 FIXED-MASK PARITY"
    );
    Serial.println(
        "======================================="
    );

    Serial.printf(
        "Vectors:                    %u\n",
        static_cast<unsigned>(
            edge_uncertainty_parity::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Passes/vector:              %u\n",
        static_cast<unsigned>(
            edge_uncertainty_parity::
                PASS_COUNT
        )
    );

    Serial.printf(
        "Prefix tensor arena:        %u / %u bytes\n",
        static_cast<unsigned>(
            inference::
                prefixRunnerTensorArenaUsedBytes()
        ),
        static_cast<unsigned>(
            inference::
                prefixRunnerTensorArenaCapacityBytes()
        )
    );

    Serial.printf(
        "B3 max abs diff:            %.9g\n",
        maxB3Diff
    );

    Serial.printf(
        "Pass probability max diff:  %.9g\n",
        maxPassProbabilityDiff
    );

    Serial.printf(
        "Mean probability max diff:  %.9g\n",
        maxMeanProbabilityDiff
    );

    Serial.printf(
        "Uncertainty max diff:       %.9g\n",
        maxUncertaintyDiff
    );

    Serial.printf(
        "Mean variance max diff:     %.9g\n",
        maxMeanVarianceDiff
    );

    Serial.printf(
        "Max variance max diff:      %.9g\n",
        maxMaxVarianceDiff
    );

    Serial.printf(
        "Top-1 agreement:            %d/%u\n",
        top1Agreement,
        static_cast<unsigned>(
            edge_uncertainty_parity::
                VECTOR_COUNT
        )
    );

    Serial.printf(
        "Acceptance tolerance B3:    %.1e\n",
        B3_TOLERANCE
    );

    Serial.printf(
        "Acceptance tolerance prob:  %.1e\n",
        PROBABILITY_TOLERANCE
    );

    Serial.printf(
        "Acceptance tolerance unc:   %.1e\n",
        UNCERTAINTY_TOLERANCE
    );

    TEST_ASSERT_LESS_OR_EQUAL_FLOAT_MESSAGE(
        B3_TOLERANCE,
        maxB3Diff,
        "ESP32 B3 prefix parity exceeded tolerance."
    );

    TEST_ASSERT_LESS_OR_EQUAL_FLOAT_MESSAGE(
        PROBABILITY_TOLERANCE,
        maxPassProbabilityDiff,
        "ESP32 masked pass probability parity exceeded tolerance."
    );

    TEST_ASSERT_LESS_OR_EQUAL_FLOAT_MESSAGE(
        PROBABILITY_TOLERANCE,
        maxMeanProbabilityDiff,
        "ESP32 mean probability parity exceeded tolerance."
    );

    TEST_ASSERT_LESS_OR_EQUAL_FLOAT_MESSAGE(
        UNCERTAINTY_TOLERANCE,
        maxUncertaintyDiff,
        "ESP32 normalized entropy parity exceeded tolerance."
    );

    TEST_ASSERT_LESS_OR_EQUAL_FLOAT_MESSAGE(
        UNCERTAINTY_TOLERANCE,
        maxMeanVarianceDiff,
        "ESP32 mean variance parity exceeded tolerance."
    );

    TEST_ASSERT_LESS_OR_EQUAL_FLOAT_MESSAGE(
        UNCERTAINTY_TOLERANCE,
        maxMaxVarianceDiff,
        "ESP32 max variance parity exceeded tolerance."
    );

    TEST_ASSERT_EQUAL_INT_MESSAGE(
        static_cast<int>(
            edge_uncertainty_parity::
                VECTOR_COUNT
        ),
        top1Agreement,
        "ESP32 mean-prediction top-1 mismatch."
    );

    Serial.println();
    Serial.println(
        "ESP32_FIXED_MASK_PARITY_PASS"
    );
}

}  // namespace


void setup() {
    delay(
        1500
    );

    Serial.begin(
        115200
    );

    delay(
        500
    );

    UNITY_BEGIN();

    RUN_TEST(
        testEsp32FixedMaskParity
    );

    UNITY_END();
}


void loop() {
}
