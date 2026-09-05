#include <Arduino.h>

#include <unity.h>

#include <cmath>
#include <cstdint>

#include "uncertainty.h"

#include "../test_edge_uncertainty_parity/edge_uncertainty_parity_vectors.h"


namespace {

constexpr size_t PROBE_VECTOR_INDEX = 2;

constexpr uint32_t DETERMINISTIC_TEST_SEED =
    0x13579BDFu;

constexpr size_t DEVICE_RUNTIME_TRIALS =
    8;

constexpr float VARIATION_TOLERANCE =
    1e-7f;


bool resultsEqual(
    const inference::UncertaintyResult& left,
    const inference::UncertaintyResult& right
) {
    for (
        size_t classIndex = 0;
        classIndex
            < inference::EDGE_HEAD_CLASS_COUNT;
        ++classIndex
    ) {
        if (
            left.meanProbabilities[
                classIndex
            ]
            != right.meanProbabilities[
                classIndex
            ]
        ) {
            return false;
        }
    }

    return (
        left.predictiveEntropyNats
            == right.predictiveEntropyNats
        && left.normalizedPredictiveEntropy
            == right.normalizedPredictiveEntropy
        && left.meanClassVariance
            == right.meanClassVariance
        && left.maxClassVariance
            == right.maxClassVariance
        && left.confidence
            == right.confidence
        && left.predictedClass
            == right.predictedClass
    );
}


void testDeterministicSeedIsReproducible() {
    const float* embedding =
        edge_uncertainty_parity::
            EXPECTED_B3[
                PROBE_VECTOR_INDEX
            ];

    inference::UncertaintyResult
        firstResult = {};

    inference::UncertaintyRuntimeDiagnostics
        firstDiagnostics = {};

    inference::seedUncertaintyMaskPrng(
        DETERMINISTIC_TEST_SEED
    );

    TEST_ASSERT_TRUE_MESSAGE(
        inference::
            runStochasticUncertaintyFromEmbedding(
                embedding,
                firstResult,
                &firstDiagnostics
            ),
        "First deterministic stochastic inference failed."
    );

    inference::UncertaintyResult
        secondResult = {};

    inference::UncertaintyRuntimeDiagnostics
        secondDiagnostics = {};

    inference::seedUncertaintyMaskPrng(
        DETERMINISTIC_TEST_SEED
    );

    TEST_ASSERT_TRUE_MESSAGE(
        inference::
            runStochasticUncertaintyFromEmbedding(
                embedding,
                secondResult,
                &secondDiagnostics
            ),
        "Second deterministic stochastic inference failed."
    );

    TEST_ASSERT_TRUE_MESSAGE(
        resultsEqual(
            firstResult,
            secondResult
        ),
        "Same PRNG seed did not reproduce the same uncertainty result."
    );

    TEST_ASSERT_EQUAL_UINT8(
        firstDiagnostics.uniqueMaskCount,
        secondDiagnostics.uniqueMaskCount
    );

    TEST_ASSERT_EQUAL_FLOAT(
        firstDiagnostics.maxPassProbabilityRange,
        secondDiagnostics.maxPassProbabilityRange
    );

    for (
        size_t passIndex = 0;
        passIndex
            < inference::UNCERTAINTY_PASS_COUNT;
        ++passIndex
    ) {
        TEST_ASSERT_EQUAL_UINT8(
            firstDiagnostics.keptUnits[
                passIndex
            ],
            secondDiagnostics.keptUnits[
                passIndex
            ]
        );
    }
}


void testDeviceSeededRuntimeActuallyVaries() {
    const float* embedding =
        edge_uncertainty_parity::
            EXPECTED_B3[
                PROBE_VECTOR_INDEX
            ];

    const uint32_t deviceSeed =
        inference::
            seedUncertaintyMaskPrngFromDevice();

    uint8_t minimumUniqueMasks =
        inference::UNCERTAINTY_PASS_COUNT;

    uint8_t minimumKeptUnits =
        inference::EDGE_HEAD_INPUT_UNITS;

    uint8_t maximumKeptUnits = 0;

    float minimumPassProbabilityRange =
        INFINITY;

    float maximumPassProbabilityRange =
        0.0f;

    float minimumUncertainty =
        INFINITY;

    float maximumUncertainty =
        0.0f;

    float firstUncertainty =
        0.0f;

    bool crossInferenceVariation =
        false;

    for (
        size_t trialIndex = 0;
        trialIndex < DEVICE_RUNTIME_TRIALS;
        ++trialIndex
    ) {
        inference::UncertaintyResult
            result = {};

        inference::UncertaintyRuntimeDiagnostics
            diagnostics = {};

        TEST_ASSERT_TRUE_MESSAGE(
            inference::
                runStochasticUncertaintyFromEmbedding(
                    embedding,
                    result,
                    &diagnostics
                ),
            "Device-seeded stochastic inference failed."
        );

        TEST_ASSERT_GREATER_OR_EQUAL_UINT8_MESSAGE(
            2,
            diagnostics.uniqueMaskCount,
            "Five passes did not produce at least two unique masks."
        );

        TEST_ASSERT_GREATER_THAN_FLOAT_MESSAGE(
            VARIATION_TOLERANCE,
            diagnostics.maxPassProbabilityRange,
            "Five passes produced no measurable probability variation."
        );

        TEST_ASSERT_TRUE_MESSAGE(
            std::isfinite(
                result.normalizedPredictiveEntropy
            ),
            "Uncertainty score is non-finite."
        );

        TEST_ASSERT_TRUE_MESSAGE(
            result.normalizedPredictiveEntropy
                >= 0.0f
            && result.normalizedPredictiveEntropy
                <= 1.0f,
            "Uncertainty score is outside [0,1]."
        );

        if (
            diagnostics.uniqueMaskCount
            < minimumUniqueMasks
        ) {
            minimumUniqueMasks =
                diagnostics.uniqueMaskCount;
        }

        if (
            diagnostics.maxPassProbabilityRange
            < minimumPassProbabilityRange
        ) {
            minimumPassProbabilityRange =
                diagnostics.maxPassProbabilityRange;
        }

        if (
            diagnostics.maxPassProbabilityRange
            > maximumPassProbabilityRange
        ) {
            maximumPassProbabilityRange =
                diagnostics.maxPassProbabilityRange;
        }

        for (
            size_t passIndex = 0;
            passIndex
                < inference::UNCERTAINTY_PASS_COUNT;
            ++passIndex
        ) {
            const uint8_t kept =
                diagnostics.keptUnits[
                    passIndex
                ];

            if (
                kept
                < minimumKeptUnits
            ) {
                minimumKeptUnits =
                    kept;
            }

            if (
                kept
                > maximumKeptUnits
            ) {
                maximumKeptUnits =
                    kept;
            }
        }

        if (
            result.normalizedPredictiveEntropy
            < minimumUncertainty
        ) {
            minimumUncertainty =
                result.normalizedPredictiveEntropy;
        }

        if (
            result.normalizedPredictiveEntropy
            > maximumUncertainty
        ) {
            maximumUncertainty =
                result.normalizedPredictiveEntropy;
        }

        if (trialIndex == 0) {
            firstUncertainty =
                result.normalizedPredictiveEntropy;
        } else if (
            std::fabs(
                result.normalizedPredictiveEntropy
                - firstUncertainty
            )
            > VARIATION_TOLERANCE
        ) {
            crossInferenceVariation =
                true;
        }
    }

    TEST_ASSERT_TRUE_MESSAGE(
        crossInferenceVariation,
        "Device-seeded PRNG sequence did not vary uncertainty across trials."
    );

    Serial.println();
    Serial.println(
        "PHASE 5 / M6 — ESP32 STOCHASTIC MASKING"
    );
    Serial.println(
        "========================================"
    );

    Serial.printf(
        "Probe validation vector:      %u\n",
        static_cast<unsigned>(
            edge_uncertainty_parity::
                VALIDATION_INDICES[
                    PROBE_VECTOR_INDEX
                ]
        )
    );

    Serial.printf(
        "Probe true class:             %d\n",
        edge_uncertainty_parity::
            TRUE_CLASSES[
                PROBE_VECTOR_INDEX
            ]
    );

    Serial.printf(
        "Device-derived PRNG seed:     0x%08lX\n",
        static_cast<unsigned long>(
            deviceSeed
        )
    );

    Serial.printf(
        "Logical inferences:           %u\n",
        static_cast<unsigned>(
            DEVICE_RUNTIME_TRIALS
        )
    );

    Serial.printf(
        "Passes/inference:             %u\n",
        static_cast<unsigned>(
            inference::
                UNCERTAINTY_PASS_COUNT
        )
    );

    Serial.printf(
        "Total generated masks:        %u\n",
        static_cast<unsigned>(
            DEVICE_RUNTIME_TRIALS
            * inference::
                UNCERTAINTY_PASS_COUNT
        )
    );

    Serial.printf(
        "Min unique masks/inference:   %u\n",
        static_cast<unsigned>(
            minimumUniqueMasks
        )
    );

    Serial.printf(
        "Kept units min/max:           %u / %u of %u\n",
        static_cast<unsigned>(
            minimumKeptUnits
        ),
        static_cast<unsigned>(
            maximumKeptUnits
        ),
        static_cast<unsigned>(
            inference::
                EDGE_HEAD_INPUT_UNITS
        )
    );

    Serial.printf(
        "Pass probability range min:   %.9g\n",
        minimumPassProbabilityRange
    );

    Serial.printf(
        "Pass probability range max:   %.9g\n",
        maximumPassProbabilityRange
    );

    Serial.printf(
        "Uncertainty score min:        %.9g\n",
        minimumUncertainty
    );

    Serial.printf(
        "Uncertainty score max:        %.9g\n",
        maximumUncertainty
    );

    Serial.println(
        "Cross-inference variation:    YES"
    );

    Serial.println(
        "Mask generator:               xorshift32"
    );

    Serial.println(
        "Seed source:                  esp_random() state"
    );

    Serial.println(
        "True-randomness claim:        NO"
    );

    Serial.println(
        "Offload threshold/policy:     NONE"
    );

    Serial.println();
    Serial.println(
        "ESP32_STOCHASTIC_MASKING_PASS"
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
        testDeterministicSeedIsReproducible
    );

    RUN_TEST(
        testDeviceSeededRuntimeActuallyVaries
    );

    UNITY_END();
}


void loop() {
}
