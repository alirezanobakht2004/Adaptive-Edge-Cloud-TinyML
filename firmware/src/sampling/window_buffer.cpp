#include "window_buffer.h"


namespace sampling {


WindowBuffer::WindowBuffer() {
    reset();
}


void WindowBuffer::reset() {
    writeIndex_ = 0;
    sampleCount_ = 0;
    samplesSinceWindow_ = 0;
    firstWindowEmitted_ = false;
}


bool WindowBuffer::pushSample(
    const float sample[SENSOR_CHANNELS]
) {
    if (sample == nullptr) {
        return false;
    }

    for (
        size_t channel = 0;
        channel < SENSOR_CHANNELS;
        ++channel
    ) {
        data_[writeIndex_][channel] =
            sample[channel];
    }

    writeIndex_ =
        (writeIndex_ + 1)
        % WINDOW_SAMPLES;

    if (sampleCount_ < WINDOW_SAMPLES) {
        ++sampleCount_;
    }

    if (sampleCount_ < WINDOW_SAMPLES) {
        return false;
    }

    if (!firstWindowEmitted_) {
        firstWindowEmitted_ = true;
        samplesSinceWindow_ = 0;

        return true;
    }

    ++samplesSinceWindow_;

    if (samplesSinceWindow_ >= STEP_SAMPLES) {
        samplesSinceWindow_ = 0;

        return true;
    }

    return false;
}


bool WindowBuffer::copyWindow(
    float output[
        WINDOW_SAMPLES
    ][
        SENSOR_CHANNELS
    ]
) const {
    if (
        output == nullptr
        || !full()
    ) {
        return false;
    }

    // Once the ring is full, writeIndex_ always points to
    // the oldest sample (the slot that will be overwritten next).
    for (
        size_t row = 0;
        row < WINDOW_SAMPLES;
        ++row
    ) {
        const size_t sourceIndex =
            (writeIndex_ + row)
            % WINDOW_SAMPLES;

        for (
            size_t channel = 0;
            channel < SENSOR_CHANNELS;
            ++channel
        ) {
            output[row][channel] =
                data_[sourceIndex][channel];
        }
    }

    return true;
}


bool WindowBuffer::full() const {
    return sampleCount_ == WINDOW_SAMPLES;
}


size_t WindowBuffer::size() const {
    return sampleCount_;
}


}  // namespace sampling
