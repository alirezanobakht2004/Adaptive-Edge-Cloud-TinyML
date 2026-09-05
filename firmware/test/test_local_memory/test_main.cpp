#include <Arduino.h>
#include <unity.h>

#include <cmath>
#include <cstdint>

#include "feature_extractor.h"
#include "gesture_model_data.h"
#include "local_model.h"
#include "window_buffer.h"


namespace {

constexpr size_t STABILITY_INVOKES = 1000;


struct MemorySnapshot {
    uint32_t heapSize;
    uint32_t freeHeap;
    uint32_t minFreeHeap;
    uint32_t maxAllocHeap;

    uint32_t psramSize;
    uint32_t freePsram;
    uint32_t maxAllocPsram;
};


MemorySnapshot captureMemory() {
    MemorySnapshot snapshot {};

    snapshot.heapSize =
        ESP.getHeapSize();

    snapshot.freeHeap =
        ESP.getFreeHeap();

    snapshot.minFreeHeap =
        ESP.getMinFreeHeap();

    snapshot.maxAllocHeap =
        ESP.getMaxAllocHeap();

    snapshot.psramSize =
        ESP.getPsramSize();

    snapshot.freePsram =
        ESP.getFreePsram();

    snapshot.maxAllocPsram =
        ESP.getMaxAllocPsram();

    return snapshot;
}


void printMemory(
    const char* label,
    const MemorySnapshot& snapshot
) {
    Serial.printf(
        "%s\n",
        label
    );

    Serial.printf(
        "  heap_size:       %lu bytes\n",
        static_cast<unsigned long>(
            snapshot.heapSize
        )
    );

    Serial.printf(
        "  free_heap:       %lu bytes\n",
        static_cast<unsigned long>(
            snapshot.freeHeap
        )
    );

    Serial.printf(
        "  min_free_heap:   %lu bytes\n",
        static_cast<unsigned long>(
            snapshot.minFreeHeap
        )
    );

    Serial.printf(
        "  max_alloc_heap:  %lu bytes\n",
        static_cast<unsigned long>(
            snapshot.maxAllocHeap
        )
    );

    Serial.printf(
        "  psram_size:      %lu bytes\n",
        static_cast<unsigned long>(
            snapshot.psramSize
        )
    );

    Serial.printf(
        "  free_psram:      %lu bytes\n",
        static_cast<unsigned long>(
            snapshot.freePsram
        )
    );

    Serial.printf(
        "  max_alloc_psram: %lu bytes\n",
        static_cast<unsigned long>(
            snapshot.maxAllocPsram
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


void testFormalLocalMemoryBenchmark() {
    const MemorySnapshot beforeInit =
        captureMemory();


    TEST_ASSERT_TRUE(
        inference::initLocalModel()
    );


    const size_t arenaCapacityBytes =
        inference::
            localModelTensorArenaCapacityBytes();

    const size_t arenaUsedBytes =
        inference::
            localModelTensorArenaUsedBytes();


    TEST_ASSERT_GREATER_THAN_UINT32(
        0,
        static_cast<uint32_t>(
            arenaCapacityBytes
        )
    );

    TEST_ASSERT_GREATER_THAN_UINT32(
        0,
        static_cast<uint32_t>(
            arenaUsedBytes
        )
    );

    TEST_ASSERT_LESS_OR_EQUAL_UINT32(
        static_cast<uint32_t>(
            arenaCapacityBytes
        ),
        static_cast<uint32_t>(
            arenaUsedBytes
        )
    );


    const MemorySnapshot afterInit =
        captureMemory();


    float input[
        inference::MODEL_INPUT_FEATURES
    ] = {};

    float output[
        inference::LOCAL_CLASS_COUNT
    ] = {};


    // Use a small deterministic non-zero vector.
    for (
        size_t i = 0;
        i < inference::MODEL_INPUT_FEATURES;
        ++i
    ) {
        input[i] =
            static_cast<float>(i)
            * 0.01f;
    }


    for (
        size_t iteration = 0;
        iteration < STABILITY_INVOKES;
        ++iteration
    ) {
        TEST_ASSERT_TRUE(
            inference::runLocalModel(
                input,
                output
            )
        );

        TEST_ASSERT_TRUE(
            probabilitiesAreFinite(
                output
            )
        );

        const int predictedClass =
            inference::localModelArgmax(
                output
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


    const MemorySnapshot afterInvokes =
        captureMemory();


    const int32_t heapDeltaInit =
        static_cast<int32_t>(
            beforeInit.freeHeap
        )
        - static_cast<int32_t>(
            afterInit.freeHeap
        );

    const int32_t heapDeltaInvokes =
        static_cast<int32_t>(
            afterInit.freeHeap
        )
        - static_cast<int32_t>(
            afterInvokes.freeHeap
        );

    const int32_t psramDeltaInit =
        static_cast<int32_t>(
            beforeInit.freePsram
        )
        - static_cast<int32_t>(
            afterInit.freePsram
        );

    const int32_t psramDeltaInvokes =
        static_cast<int32_t>(
            afterInit.freePsram
        )
        - static_cast<int32_t>(
            afterInvokes.freePsram
        );


    const size_t arenaHeadroomBytes =
        arenaCapacityBytes
        - arenaUsedBytes;

    const double arenaUtilizationPercent =
        100.0
        * static_cast<double>(
            arenaUsedBytes
        )
        / static_cast<double>(
            arenaCapacityBytes
        );


    constexpr size_t chronologicalWindowBytes =
        sampling::WINDOW_SAMPLES
        * sampling::SENSOR_CHANNELS
        * sizeof(float);

    constexpr size_t featureVectorBytes =
        features::FEATURE_COUNT
        * sizeof(float);

    constexpr size_t normalizedInputBytes =
        inference::MODEL_INPUT_FEATURES
        * sizeof(float);

    constexpr size_t probabilityBytes =
        inference::LOCAL_CLASS_COUNT
        * sizeof(float);


    Serial.println();
    Serial.println(
        "FORMAL LOCAL MEMORY BENCHMARK"
    );
    Serial.println(
        "============================="
    );

    Serial.printf(
        "Model blob:                  %lu bytes\n",
        static_cast<unsigned long>(
            gesture_model_data::MODEL_LEN
        )
    );

    Serial.printf(
        "Tensor arena capacity:       %lu bytes\n",
        static_cast<unsigned long>(
            arenaCapacityBytes
        )
    );

    Serial.printf(
        "Tensor arena used:           %lu bytes\n",
        static_cast<unsigned long>(
            arenaUsedBytes
        )
    );

    Serial.printf(
        "Tensor arena headroom:       %lu bytes\n",
        static_cast<unsigned long>(
            arenaHeadroomBytes
        )
    );

    Serial.printf(
        "Tensor arena utilization:    %.2f %%\n",
        arenaUtilizationPercent
    );

    Serial.println();

    Serial.printf(
        "WindowBuffer object:         %lu bytes\n",
        static_cast<unsigned long>(
            sizeof(sampling::WindowBuffer)
        )
    );

    Serial.printf(
        "Chronological window copy:   %lu bytes\n",
        static_cast<unsigned long>(
            chronologicalWindowBytes
        )
    );

    Serial.printf(
        "Feature vector:              %lu bytes\n",
        static_cast<unsigned long>(
            featureVectorBytes
        )
    );

    Serial.printf(
        "Normalized model input:      %lu bytes\n",
        static_cast<unsigned long>(
            normalizedInputBytes
        )
    );

    Serial.printf(
        "Probability output:          %lu bytes\n",
        static_cast<unsigned long>(
            probabilityBytes
        )
    );

    Serial.println();

    printMemory(
        "BEFORE MODEL INIT",
        beforeInit
    );

    printMemory(
        "AFTER MODEL INIT",
        afterInit
    );

    printMemory(
        "AFTER 1000 INVOKES",
        afterInvokes
    );

    Serial.println();

    Serial.printf(
        "Heap delta during init:      %ld bytes\n",
        static_cast<long>(
            heapDeltaInit
        )
    );

    Serial.printf(
        "Heap delta after invokes:    %ld bytes\n",
        static_cast<long>(
            heapDeltaInvokes
        )
    );

    Serial.printf(
        "PSRAM delta during init:     %ld bytes\n",
        static_cast<long>(
            psramDeltaInit
        )
    );

    Serial.printf(
        "PSRAM delta after invokes:   %ld bytes\n",
        static_cast<long>(
            psramDeltaInvokes
        )
    );

    Serial.println();
    Serial.println(
        "NOTE: the production Tensor Arena is a static buffer,"
    );

    Serial.println(
        "so its 64 KiB capacity is reflected in static RAM/build"
    );

    Serial.println(
        "footprint rather than in the runtime heap delta above."
    );

    Serial.println(
        "MEMORY_BENCHMARK_COMPLETE"
    );


    TEST_ASSERT_GREATER_THAN_UINT32(
        0,
        gesture_model_data::MODEL_LEN
    );

    TEST_ASSERT_EQUAL_UINT32(
        sampling::WINDOW_SAMPLES
            * sampling::SENSOR_CHANNELS
            * sizeof(float),
        chronologicalWindowBytes
    );
}

}  // namespace


void setup() {
    delay(2000);

    Serial.begin(115200);

    delay(500);

    UNITY_BEGIN();

    RUN_TEST(
        testFormalLocalMemoryBenchmark
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
