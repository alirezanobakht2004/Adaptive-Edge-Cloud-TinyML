#include "attitude_estimator.h"

#include <cmath>

namespace attitude {
namespace {

constexpr float RADIANS_TO_DEGREES = 57.29577951308232f;
constexpr float COMPLEMENTARY_ALPHA = 0.95f;
constexpr float MIN_VALID_DT_SECONDS = 0.001f;
constexpr float MAX_VALID_DT_SECONDS = 0.100f;
constexpr float MIN_GRAVITY_GATE_G = 0.75f;
constexpr float MAX_GRAVITY_GATE_G = 1.25f;

bool finiteSample(const ImuSample& sample) {
    return std::isfinite(sample.ax)
        && std::isfinite(sample.ay)
        && std::isfinite(sample.az)
        && std::isfinite(sample.gx)
        && std::isfinite(sample.gy)
        && std::isfinite(sample.gz);
}

float wrapDegrees(float value) {
    while (value > 180.0f) value -= 360.0f;
    while (value <= -180.0f) value += 360.0f;
    return value;
}

bool accelerometerAngles(const ImuSample& sample, float& rollDeg, float& pitchDeg, float& magnitudeG) {
    const float magnitudeSquared = sample.ax * sample.ax + sample.ay * sample.ay + sample.az * sample.az;
    if (!std::isfinite(magnitudeSquared) || magnitudeSquared < 1.0e-6f) return false;

    magnitudeG = std::sqrt(magnitudeSquared);
    rollDeg = std::atan2(sample.ay, sample.az) * RADIANS_TO_DEGREES;
    pitchDeg = std::atan2(-sample.ax, std::sqrt(sample.ay * sample.ay + sample.az * sample.az)) * RADIANS_TO_DEGREES;
    return std::isfinite(rollDeg) && std::isfinite(pitchDeg) && std::isfinite(magnitudeG);
}

}  // namespace

void ComplementaryAttitudeEstimator::reset() {
    initialized_ = false;
    lastTimestampUs_ = 0;
    sequence_ = 0;
    rollDeg_ = 0.0f;
    pitchDeg_ = 0.0f;
    yawRelativeDeg_ = 0.0f;
}

bool ComplementaryAttitudeEstimator::update(
    const ImuSample& sample,
    uint32_t sampleTimestampUs,
    uint32_t timestampMs,
    PoseEstimate& out
) {
    if (!finiteSample(sample)) return false;

    float accelRollDeg = 0.0f;
    float accelPitchDeg = 0.0f;
    float accelMagnitudeG = 0.0f;
    const bool accelValid = accelerometerAngles(sample, accelRollDeg, accelPitchDeg, accelMagnitudeG);

    if (!initialized_) {
        if (!accelValid) return false;
        rollDeg_ = accelRollDeg;
        pitchDeg_ = accelPitchDeg;
        yawRelativeDeg_ = 0.0f;
        lastTimestampUs_ = sampleTimestampUs;
        initialized_ = true;
    } else {
        const uint32_t deltaUs = sampleTimestampUs - lastTimestampUs_;
        lastTimestampUs_ = sampleTimestampUs;
        const float dtSeconds = static_cast<float>(deltaUs) / 1000000.0f;

        // Do not integrate across an unexpected long scheduler/I2C pause. This avoids
        // turning a timing discontinuity into a fabricated orientation jump.
        if (dtSeconds >= MIN_VALID_DT_SECONDS && dtSeconds <= MAX_VALID_DT_SECONDS) {
            const float gyroRoll = rollDeg_ + sample.gx * dtSeconds;
            const float gyroPitch = pitchDeg_ + sample.gy * dtSeconds;
            yawRelativeDeg_ = wrapDegrees(yawRelativeDeg_ + sample.gz * dtSeconds);

            const bool gravityTrustworthy = accelValid
                && accelMagnitudeG >= MIN_GRAVITY_GATE_G
                && accelMagnitudeG <= MAX_GRAVITY_GATE_G;

            if (gravityTrustworthy) {
                rollDeg_ = COMPLEMENTARY_ALPHA * gyroRoll
                    + (1.0f - COMPLEMENTARY_ALPHA) * accelRollDeg;
                pitchDeg_ = COMPLEMENTARY_ALPHA * gyroPitch
                    + (1.0f - COMPLEMENTARY_ALPHA) * accelPitchDeg;
            } else {
                rollDeg_ = gyroRoll;
                pitchDeg_ = gyroPitch;
            }
        }
    }

    rollDeg_ = wrapDegrees(rollDeg_);
    pitchDeg_ = wrapDegrees(pitchDeg_);

    if (!std::isfinite(rollDeg_) || !std::isfinite(pitchDeg_) || !std::isfinite(yawRelativeDeg_)) {
        reset();
        return false;
    }

    ++sequence_;
    out.sequence = sequence_;
    out.timestampMs = timestampMs;
    out.rollDeg = wrapDegrees(rollDeg_);
    out.pitchDeg = pitchDeg_;
    out.yawRelativeDeg = wrapDegrees(yawRelativeDeg_);
    return true;
}

}  // namespace attitude
