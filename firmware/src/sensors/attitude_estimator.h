#pragma once

#include <Arduino.h>

#include "mpu6050.h"

namespace attitude {

constexpr const char* ESTIMATOR_VERSION = "attitude-complementary-v1";
constexpr const char* POSE_SCHEMA_VERSION = "pose-v1";
constexpr const char* POSE_SOURCE = "mpu6050-6axis";
constexpr const char* YAW_REFERENCE = "boot-relative";

struct PoseEstimate {
    uint32_t sequence = 0;
    uint32_t timestampMs = 0;
    float rollDeg = 0.0f;
    float pitchDeg = 0.0f;
    float yawRelativeDeg = 0.0f;
};

class ComplementaryAttitudeEstimator {
public:
    void reset();

    // Updates the estimator from the already-acquired 100 Hz IMU sample.
    // Returns true when a finite pose estimate is available in `out`.
    bool update(
        const ImuSample& sample,
        uint32_t sampleTimestampUs,
        uint32_t timestampMs,
        PoseEstimate& out
    );

private:
    bool initialized_ = false;
    uint32_t lastTimestampUs_ = 0;
    uint32_t sequence_ = 0;
    float rollDeg_ = 0.0f;
    float pitchDeg_ = 0.0f;
    float yawRelativeDeg_ = 0.0f;
};

}  // namespace attitude
