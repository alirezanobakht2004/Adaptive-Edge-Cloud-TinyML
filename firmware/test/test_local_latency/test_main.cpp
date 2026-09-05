#include <Arduino.h>
#include <unity.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "feature_extractor.h"
#include "input_preprocessor.h"
#include "local_model.h"

#include "../test_feature_parity/feature_parity_vectors.h"


namespace {

constexpr size_t WARMUP_ITERATIONS = 50;
constexpr size_t BENCHMARK_ITERATIONS = 1000;

static_assert(
    feature_parity::VECTOR_COUNT == 5,
    "Latency benchmark expects five validation vectors."
);

static_assert(
    feature_parity::WINDOW_SAMPLES
        == features::WINDOW_SAMPLES,
    "Window sample count mismatch."
);

static_assert(
    feature_parity::SENSOR_CHANNELS
        == features::SENSOR_CHANNELS,
    "Sensor channel count mismatch."
);

static_assert(
    features::FEATURE_COUNT
        == inference::MODEL_INPUT_FEATURES,
    "Feature count must match model input."
);


uint32_t featureTimesUs[BENCHMARK_ITERATIONS];
uint32_t normalizeTimesUs[BENCHMARK_ITERATIONS];
uint32_t invokeTimesUs[BENCHMARK_ITERATIONS];
uint32_t pipelineTimesUs[BENCHMARK_ITERATIONS];


struct TimingStats {
    uint32_t minUs;
    double meanUs;
    double medianUs;
    uint32_t p95Us;
    uint32_t maxUs;
};


TimingStats computeStats(
    uint32_t values[BENCHMARK_ITERATIONS]
) {
    std::sort(
        values,
        values + BENCHMARK_ITERATIONS
    );

    uint64_t sum = 0;

    for (
        size_t i = 0;
        i < BENCHMARK_ITERATIONS;
        ++i
    ) {
        sum += values[i];
    }

    constexpr size_t p95Index =
        (
            95 * BENCHMARK_ITERATIONS
            + 99
        )
        / 100
        - 1;

    TimingStats stats {};

    stats.minUs =
        values[0];

    stats.meanUs =
        static_cast<double>(sum)
        / static_cast<double>(
            BENCHMARK_ITERATIONS
        );

    if (
        BENCHMARK_ITERATIONS % 2
        == 0
    ) {
        stats.medianUs =
            (
                static_cast<double>(
                    values[
                        BENCHMARK_ITERATIONS / 2 - 1
                    ]
                )
                + static_cast<double>(
                    values[
                        BENCHMARK_ITERATIONS / 2
                    ]
                )
            )
            / 2.0;
    } else {
        stats.medianUs =
            static_cast<double>(
                values[
                    BENCHMARK_ITERATIONS / 2
                ]
            );
    }

    stats.p95Us =
        values[p95Index];

    stats.maxUs =
        values[
            BENCHMARK_ITERATIONS - 1
        ];

    return stats;
}


void printStats(
    const char* name,
    const TimingStats& stats
) {
    Serial.printf(
        "%-18s "
        "min=%lu us "
        "mean=%.2f us "
        "median=%.2f us "
        "p95=%lu us "
        "max=%lu us\n",
        name,
        static_cast<unsigned long>(
            stats.minUs
        ),
        stats.meanUs,
        stats.medianUs,
        static_cast<unsigned long>(
            stats.p95Us
        ),
        static_cast<unsigned long>(
            stats.maxUs
        )
    );
}


bool probabilitiesAreFinite(
    const float probabilities[
        inference::LOCAL_CLASS_COUNT
    ]
) {
    for (
        size_t i = 0;
        i < inference::LOCAL_CLASS_COUNT;
        ++i
    ) {
        if (!std::isfinite(probabilities[i])) {
            return false;
        }
    }

    return true;
}


void testFormalLocalLatencyBenchmark() {
    TEST_ASSERT_TRUE(
        inference::initLocalModel()
    );

    float featuresVector[
        features::FEATURE_COUNT
    ] = {};

    float normalized[
        inference::MODEL_INPUT_FEATURES
    ] = {};

    float probabilities[
        inference::LOCAL_CLASS_COUNT
    ] = {};


    // Measure the first model invocation separately.
    // This preserves the cold/first-invoke behavior seen in live runtime.
    TEST_ASSERT_TRUE(
        features::extractFeaturesV1(
            feature_parity::WINDOWS[0],
            featuresVector
        )
    );

    TEST_ASSERT_TRUE(
        inference::normalizeFeaturesV1(
            featuresVector,
            normalized
        )
    );

    const uint32_t coldInvokeStartUs =
        micros();

    TEST_ASSERT_TRUE(
        inference::runLocalModel(
            normalized,
            probabilities
        )
    );

    const uint32_t coldInvokeUs =
        micros()
        - coldInvokeStartUs;

    TEST_ASSERT_TRUE(
        probabilitiesAreFinite(
            probabilities
        )
    );


    // Warm up the exact production compute path.
    for (
        size_t iteration = 0;
        iteration < WARMUP_ITERATIONS;
        ++iteration
    ) {
        const size_t vectorIndex =
            iteration
            % feature_parity::VECTOR_COUNT;

        TEST_ASSERT_TRUE(
            features::extractFeaturesV1(
                feature_parity::WINDOWS[
                    vectorIndex
                ],
                featuresVector
            )
        );

        TEST_ASSERT_TRUE(
            inference::normalizeFeaturesV1(
                featuresVector,
                normalized
            )
        );

        TEST_ASSERT_TRUE(
            inference::runLocalModel(
                normalized,
                probabilities
            )
        );

        TEST_ASSERT_TRUE(
            probabilitiesAreFinite(
                probabilities
            )
        );

        const int predictedClass =
            inference::localModelArgmax(
                probabilities
            );

        TEST_ASSERT_GREATER_OR_EQUAL_INT(
            0,
            predictedClass
        );

        TEST_ASSERT_LESS_THAN_INT(
            static_cast<int>(
                inference::LOCAL_CLASS_COUNT
            ),
            predictedClass
        );
    }


    volatile float outputChecksum = 0.0f;


    // Formal steady-state measurement.
    // No Serial output occurs inside this loop.
    for (
        size_t iteration = 0;
        iteration < BENCHMARK_ITERATIONS;
        ++iteration
    ) {
        const size_t vectorIndex =
            iteration
            % feature_parity::VECTOR_COUNT;


        const uint32_t pipelineStartUs =
            micros();

        const uint32_t featureStartUs =
            pipelineStartUs;

        const bool featureOk =
            features::extractFeaturesV1(
                feature_parity::WINDOWS[
                    vectorIndex
                ],
                featuresVector
            );

        const uint32_t featureEndUs =
            micros();


        const bool normalizeOk =
            inference::normalizeFeaturesV1(
                featuresVector,
                normalized
            );

        const uint32_t normalizeEndUs =
            micros();


        const bool invokeOk =
            inference::runLocalModel(
                normalized,
                probabilities
            );

        const uint32_t invokeEndUs =
            micros();


        const int predictedClass =
            inference::localModelArgmax(
                probabilities
            );

        const uint32_t pipelineEndUs =
            micros();


        TEST_ASSERT_TRUE(featureOk);
        TEST_ASSERT_TRUE(normalizeOk);
        TEST_ASSERT_TRUE(invokeOk);

        TEST_ASSERT_TRUE(
            probabilitiesAreFinite(
                probabilities
            )
        );

        TEST_ASSERT_GREATER_OR_EQUAL_INT(
            0,
            predictedClass
        );

        TEST_ASSERT_LESS_THAN_INT(
            static_cast<int>(
                inference::LOCAL_CLASS_COUNT
            ),
            predictedClass
        );


        featureTimesUs[iteration] =
            featureEndUs
            - featureStartUs;

        normalizeTimesUs[iteration] =
            normalizeEndUs
            - featureEndUs;

        invokeTimesUs[iteration] =
            invokeEndUs
            - normalizeEndUs;

        pipelineTimesUs[iteration] =
            pipelineEndUs
            - pipelineStartUs;


        outputChecksum +=
            probabilities[
                static_cast<size_t>(
                    predictedClass
                )
            ];
    }


    TimingStats featureStats =
        computeStats(
            featureTimesUs
        );

    TimingStats normalizeStats =
        computeStats(
            normalizeTimesUs
        );

    TimingStats invokeStats =
        computeStats(
            invokeTimesUs
        );

    TimingStats pipelineStats =
        computeStats(
            pipelineTimesUs
        );


    Serial.println();
    Serial.println(
        "FORMAL LOCAL LATENCY BENCHMARK"
    );
    Serial.println(
        "=============================="
    );

    Serial.printf(
        "Vectors:              %u validation windows\n",
        static_cast<unsigned>(
            feature_parity::VECTOR_COUNT
        )
    );

    Serial.printf(
        "Warmup iterations:    %u\n",
        static_cast<unsigned>(
            WARMUP_ITERATIONS
        )
    );

    Serial.printf(
        "Measured iterations:  %u\n",
        static_cast<unsigned>(
            BENCHMARK_ITERATIONS
        )
    );

    Serial.println(
        "Serial logging inside measured loop: NO"
    );

    Serial.println(
        "Sensor acquisition included: NO"
    );

    Serial.println(
        "WindowBuffer copy included: NO"
    );

    Serial.println(
        "Pipeline = features + normalization + Float32 TFLM + argmax"
    );

    Serial.printf(
        "First/cold invoke:     %lu us\n",
        static_cast<unsigned long>(
            coldInvokeUs
        )
    );

    Serial.println();

    printStats(
        "Feature extraction",
        featureStats
    );

    printStats(
        "Normalization",
        normalizeStats
    );

    printStats(
        "TFLM invoke",
        invokeStats
    );

    printStats(
        "Compute pipeline",
        pipelineStats
    );

    Serial.printf(
        "Checksum:              %.6f\n",
        static_cast<double>(
            outputChecksum
        )
    );

    Serial.println(
        "LATENCY_BENCHMARK_COMPLETE"
    );


    TEST_ASSERT_GREATER_THAN_UINT32(
        0,
        coldInvokeUs
    );

    TEST_ASSERT_GREATER_THAN_UINT32(
        0,
        featureStats.maxUs
    );

    TEST_ASSERT_GREATER_THAN_UINT32(
        0,
        invokeStats.maxUs
    );

    TEST_ASSERT_GREATER_THAN_UINT32(
        0,
        pipelineStats.maxUs
    );
}

}  // namespace


void setup() {
    delay(2000);

    Serial.begin(115200);

    delay(500);

    UNITY_BEGIN();

    RUN_TEST(
        testFormalLocalLatencyBenchmark
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
