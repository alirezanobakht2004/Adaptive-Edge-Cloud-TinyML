#pragma once

#include <stddef.h>

#include "feature_extractor.h"


namespace sampling {

constexpr size_t WINDOW_SAMPLES =
    features::WINDOW_SAMPLES;

constexpr size_t SENSOR_CHANNELS =
    features::SENSOR_CHANNELS;

constexpr size_t STEP_SAMPLES =
    WINDOW_SAMPLES / 2;

static_assert(
    WINDOW_SAMPLES == 100,
    "Runtime window contract expects 100 samples."
);

static_assert(
    STEP_SAMPLES == 50,
    "Runtime step contract expects 50 samples."
);


class WindowBuffer {
public:
    WindowBuffer();

    void reset();

    // Adds one 6-channel IMU sample.
    //
    // Returns true only when this sample completes a new inference window:
    // - first time at total sample 100
    // - then every 50 new samples
    //
    // The caller should call copyWindow() immediately when true is returned.
    bool pushSample(
        const float sample[SENSOR_CHANNELS]
    );

    // Copies the current 100-sample window in chronological order
    // from oldest sample to newest sample.
    //
    // Returns false until the buffer has received 100 samples.
    bool copyWindow(
        float output[WINDOW_SAMPLES][SENSOR_CHANNELS]
    ) const;

    bool full() const;

    size_t size() const;

private:
    float data_[
        WINDOW_SAMPLES
    ][
        SENSOR_CHANNELS
    ];

    size_t writeIndex_;
    size_t sampleCount_;
    size_t samplesSinceWindow_;
    bool firstWindowEmitted_;
};


}  // namespace sampling
