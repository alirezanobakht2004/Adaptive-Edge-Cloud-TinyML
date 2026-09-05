#include "feature_extractor.h"
#include <cmath>

namespace features {

namespace {

struct RunningStats {
    size_t count = 0;
    float mean = 0.0f;
    float m2 = 0.0f;

    void add(float value) {
        count++;

        const float delta = value - mean;

        mean += delta / static_cast<float>(count);

        const float delta2 = value - mean;

        m2 += delta * delta2;
    }

    float populationStd() const {
        if (count == 0) {
            return 0.0f;
        }

        const float variance =
            m2 / static_cast<float>(count);

        return std::sqrt(
            variance > 0.0f ? variance : 0.0f
        );
    }
};

bool finiteValue(float value) {
    return std::isfinite(value);
}

}  // namespace


bool extractFeaturesV1(
    const float window[WINDOW_SAMPLES][SENSOR_CHANNELS],
    float output[FEATURE_COUNT]
) {
    if (window == nullptr || output == nullptr) {
        return false;
    }

    RunningStats axStats;
    RunningStats ayStats;
    RunningStats azStats;
    RunningStats gzStats;
    RunningStats accMagnitudeStats;

    float maxAbsAx = 0.0f;

    float firstHalfAxSum = 0.0f;
    float secondHalfAxSum = 0.0f;

    float gyroMagnitudeSquaredSum = 0.0f;
    float maxGyroMagnitudeSquared = 0.0f;

    constexpr size_t HALF_WINDOW =
        WINDOW_SAMPLES / 2;

    for (size_t index = 0;
         index < WINDOW_SAMPLES;
         ++index) {

        const float ax = window[index][0];
        const float ay = window[index][1];
        const float az = window[index][2];

        const float gx = window[index][3];
        const float gy = window[index][4];
        const float gz = window[index][5];

        if (
            !finiteValue(ax)
            || !finiteValue(ay)
            || !finiteValue(az)
            || !finiteValue(gx)
            || !finiteValue(gy)
            || !finiteValue(gz)
        ) {
            return false;
        }

        axStats.add(ax);
        ayStats.add(ay);
        azStats.add(az);
        gzStats.add(gz);

        const float absAx = std::fabs(ax);

        if (absAx > maxAbsAx) {
            maxAbsAx = absAx;
        }

        if (index < HALF_WINDOW) {
            firstHalfAxSum += ax;
        } else {
            secondHalfAxSum += ax;
        }

        const float accMagnitude = std::sqrt(
            ax * ax
            + ay * ay
            + az * az
        );

        accMagnitudeStats.add(
            accMagnitude
        );

        const float gyroMagnitudeSquared =
            gx * gx
            + gy * gy
            + gz * gz;

        gyroMagnitudeSquaredSum +=
            gyroMagnitudeSquared;

        if (
            gyroMagnitudeSquared
            > maxGyroMagnitudeSquared
        ) {
            maxGyroMagnitudeSquared =
                gyroMagnitudeSquared;
        }
    }

    const float firstHalfAxMean =
        firstHalfAxSum
        / static_cast<float>(HALF_WINDOW);

    const float secondHalfAxMean =
        secondHalfAxSum
        / static_cast<float>(HALF_WINDOW);

    output[0] = axStats.populationStd();

    output[1] = maxAbsAx;

    output[2] =
        firstHalfAxMean
        - secondHalfAxMean;

    output[3] = ayStats.populationStd();

    output[4] = azStats.populationStd();

    output[5] =
        accMagnitudeStats.populationStd();

    output[6] = gzStats.mean;

    output[7] = gzStats.populationStd();

    output[8] = std::sqrt(
        gyroMagnitudeSquaredSum
        / static_cast<float>(WINDOW_SAMPLES)
    );

    output[9] = std::sqrt(
        maxGyroMagnitudeSquared
    );

    for (size_t index = 0;
         index < FEATURE_COUNT;
         ++index) {

        if (!finiteValue(output[index])) {
            return false;
        }
    }

    return true;
}

}  // namespace features