#include <Arduino.h>
#include <unity.h>

#include "window_buffer.h"


namespace {

float windowA[
    sampling::WINDOW_SAMPLES
][
    sampling::SENSOR_CHANNELS
];

float windowB[
    sampling::WINDOW_SAMPLES
][
    sampling::SENSOR_CHANNELS
];


void makeSample(
    size_t sampleIndex,
    float output[sampling::SENSOR_CHANNELS]
) {
    for (
        size_t channel = 0;
        channel < sampling::SENSOR_CHANNELS;
        ++channel
    ) {
        output[channel] =
            static_cast<float>(
                sampleIndex * 10 + channel
            );
    }
}


void assertWindowContainsRange(
    const float window[
        sampling::WINDOW_SAMPLES
    ][
        sampling::SENSOR_CHANNELS
    ],
    size_t firstSampleIndex
) {
    for (
        size_t row = 0;
        row < sampling::WINDOW_SAMPLES;
        ++row
    ) {
        const size_t expectedSampleIndex =
            firstSampleIndex + row;

        for (
            size_t channel = 0;
            channel < sampling::SENSOR_CHANNELS;
            ++channel
        ) {
            const float expected =
                static_cast<float>(
                    expectedSampleIndex * 10
                    + channel
                );

            TEST_ASSERT_EQUAL_FLOAT(
                expected,
                window[row][channel]
            );
        }
    }
}


void testContractConstants() {
    TEST_ASSERT_EQUAL_UINT32(
        100,
        sampling::WINDOW_SAMPLES
    );

    TEST_ASSERT_EQUAL_UINT32(
        50,
        sampling::STEP_SAMPLES
    );

    TEST_ASSERT_EQUAL_UINT32(
        6,
        sampling::SENSOR_CHANNELS
    );
}


void testFirstWindowAppearsAtSample100() {
    sampling::WindowBuffer buffer;

    float sample[
        sampling::SENSOR_CHANNELS
    ];

    TEST_ASSERT_FALSE(buffer.full());
    TEST_ASSERT_EQUAL_UINT32(
        0,
        buffer.size()
    );

    for (
        size_t sampleIndex = 0;
        sampleIndex < 99;
        ++sampleIndex
    ) {
        makeSample(
            sampleIndex,
            sample
        );

        TEST_ASSERT_FALSE(
            buffer.pushSample(sample)
        );
    }

    TEST_ASSERT_FALSE(buffer.full());
    TEST_ASSERT_EQUAL_UINT32(
        99,
        buffer.size()
    );

    makeSample(
        99,
        sample
    );

    TEST_ASSERT_TRUE(
        buffer.pushSample(sample)
    );

    TEST_ASSERT_TRUE(buffer.full());

    TEST_ASSERT_EQUAL_UINT32(
        100,
        buffer.size()
    );

    TEST_ASSERT_TRUE(
        buffer.copyWindow(windowA)
    );

    assertWindowContainsRange(
        windowA,
        0
    );
}


void testNextWindowAppearsAfter50NewSamples() {
    sampling::WindowBuffer buffer;

    float sample[
        sampling::SENSOR_CHANNELS
    ];

    for (
        size_t sampleIndex = 0;
        sampleIndex < 100;
        ++sampleIndex
    ) {
        makeSample(
            sampleIndex,
            sample
        );

        const bool ready =
            buffer.pushSample(sample);

        if (sampleIndex < 99) {
            TEST_ASSERT_FALSE(ready);
        } else {
            TEST_ASSERT_TRUE(ready);
        }
    }

    TEST_ASSERT_TRUE(
        buffer.copyWindow(windowA)
    );

    for (
        size_t sampleIndex = 100;
        sampleIndex < 149;
        ++sampleIndex
    ) {
        makeSample(
            sampleIndex,
            sample
        );

        TEST_ASSERT_FALSE(
            buffer.pushSample(sample)
        );
    }

    makeSample(
        149,
        sample
    );

    TEST_ASSERT_TRUE(
        buffer.pushSample(sample)
    );

    TEST_ASSERT_TRUE(
        buffer.copyWindow(windowB)
    );

    // Window A = samples 0..99
    // Window B = samples 50..149
    assertWindowContainsRange(
        windowA,
        0
    );

    assertWindowContainsRange(
        windowB,
        50
    );

    // The last 50 samples of the first window must be
    // exactly the first 50 samples of the second window.
    for (
        size_t row = 0;
        row < sampling::STEP_SAMPLES;
        ++row
    ) {
        for (
            size_t channel = 0;
            channel < sampling::SENSOR_CHANNELS;
            ++channel
        ) {
            TEST_ASSERT_EQUAL_FLOAT(
                windowA[
                    row + sampling::STEP_SAMPLES
                ][channel],
                windowB[row][channel]
            );
        }
    }
}


void testRingWrapPreservesChronologicalOrder() {
    sampling::WindowBuffer buffer;

    float sample[
        sampling::SENSOR_CHANNELS
    ];

    size_t readyCount = 0;

    for (
        size_t sampleIndex = 0;
        sampleIndex < 200;
        ++sampleIndex
    ) {
        makeSample(
            sampleIndex,
            sample
        );

        if (buffer.pushSample(sample)) {
            ++readyCount;
        }
    }

    // Ready at total samples 100, 150, and 200.
    TEST_ASSERT_EQUAL_UINT32(
        3,
        readyCount
    );

    TEST_ASSERT_TRUE(
        buffer.copyWindow(windowA)
    );

    // Latest 100 samples after 200 pushes are 100..199.
    assertWindowContainsRange(
        windowA,
        100
    );
}


void testResetRestoresEmptyState() {
    sampling::WindowBuffer buffer;

    float sample[
        sampling::SENSOR_CHANNELS
    ];

    for (
        size_t sampleIndex = 0;
        sampleIndex < 100;
        ++sampleIndex
    ) {
        makeSample(
            sampleIndex,
            sample
        );

        buffer.pushSample(sample);
    }

    TEST_ASSERT_TRUE(buffer.full());

    buffer.reset();

    TEST_ASSERT_FALSE(buffer.full());

    TEST_ASSERT_EQUAL_UINT32(
        0,
        buffer.size()
    );

    TEST_ASSERT_FALSE(
        buffer.copyWindow(windowA)
    );
}


}  // namespace


void setup() {
    delay(2000);

    Serial.begin(115200);

    delay(500);

    UNITY_BEGIN();

    RUN_TEST(
        testContractConstants
    );

    RUN_TEST(
        testFirstWindowAppearsAtSample100
    );

    RUN_TEST(
        testNextWindowAppearsAfter50NewSamples
    );

    RUN_TEST(
        testRingWrapPreservesChronologicalOrder
    );

    RUN_TEST(
        testResetRestoresEmptyState
    );

    UNITY_END();
}


void loop() {
    delay(1000);
}
