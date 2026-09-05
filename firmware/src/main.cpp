#ifndef PIO_UNIT_TESTING

#include <Arduino.h>
#include <Wire.h>

#include <climits>

#include "version.h"
#include "sensors/mpu6050.h"
#include "feature_extractor.h"
#include "input_preprocessor.h"
#include "local_model.h"
#include "window_buffer.h"


namespace {

constexpr uint8_t SDA_PIN = 8;
constexpr uint8_t SCL_PIN = 9;

constexpr uint32_t I2C_FREQUENCY_HZ = 400000;

constexpr uint32_t SAMPLE_RATE_HZ = 100;
constexpr uint32_t SAMPLE_PERIOD_US =
    1000000UL / SAMPLE_RATE_HZ;

constexpr uint16_t GYRO_CALIBRATION_SAMPLES = 200;


static_assert(
    sampling::WINDOW_SAMPLES == 100,
    "Continuous runtime expects 100-sample windows."
);

static_assert(
    sampling::STEP_SAMPLES == 50,
    "Continuous runtime expects 50-sample steps."
);

static_assert(
    sampling::SENSOR_CHANNELS == 6,
    "Continuous runtime expects six IMU channels."
);

static_assert(
    features::FEATURE_COUNT
        == inference::MODEL_INPUT_FEATURES,
    "Feature count must match model input."
);


sampling::WindowBuffer runtimeBuffer;

float inferenceWindow
    [sampling::WINDOW_SAMPLES]
    [sampling::SENSOR_CHANNELS];


uint32_t nextSampleUs = 0;
uint32_t previousSampleUs = 0;
uint32_t previousWindowReadyUs = 0;

uint32_t successfulSamples = 0;
uint32_t readFailures = 0;
uint32_t fullPeriodOverruns = 0;
uint32_t windowCount = 0;

uint64_t periodSumUs = 0;
uint32_t periodCount = 0;
uint32_t minPeriodUs = UINT32_MAX;
uint32_t maxPeriodUs = 0;


[[noreturn]]
void fatal(const char* message) {
    Serial.printf(
        "FATAL: %s\n",
        message
    );

    while (true) {
        delay(1000);
    }
}


void resetPeriodStats() {
    periodSumUs = 0;
    periodCount = 0;
    minPeriodUs = UINT32_MAX;
    maxPeriodUs = 0;
}


void updatePeriodStats(
    uint32_t sampleTimestampUs
) {
    if (previousSampleUs != 0) {
        const uint32_t periodUs =
            sampleTimestampUs
            - previousSampleUs;

        periodSumUs += periodUs;
        ++periodCount;

        if (periodUs < minPeriodUs) {
            minPeriodUs = periodUs;
        }

        if (periodUs > maxPeriodUs) {
            maxPeriodUs = periodUs;
        }
    }

    previousSampleUs = sampleTimestampUs;
}


void runWindowInference(
    uint32_t windowReadyUs
) {
    if (
        !runtimeBuffer.copyWindow(
            inferenceWindow
        )
    ) {
        fatal(
            "window copy failed."
        );
    }


    float featureVector[
        features::FEATURE_COUNT
    ] = {};

    float normalized[
        inference::MODEL_INPUT_FEATURES
    ] = {};

    float probabilities[
        inference::LOCAL_CLASS_COUNT
    ] = {};


    const uint32_t pipelineStartUs =
        micros();

    const uint32_t featureStartUs =
        pipelineStartUs;

    if (
        !features::extractFeaturesV1(
            inferenceWindow,
            featureVector
        )
    ) {
        fatal(
            "features-v1 extraction failed."
        );
    }

    const uint32_t featureEndUs =
        micros();


    if (
        !inference::normalizeFeaturesV1(
            featureVector,
            normalized
        )
    ) {
        fatal(
            "feature normalization failed."
        );
    }

    const uint32_t normalizeEndUs =
        micros();


    if (
        !inference::runLocalModel(
            normalized,
            probabilities
        )
    ) {
        fatal(
            "local Float32 inference failed."
        );
    }

    const uint32_t invokeEndUs =
        micros();


    const int predictedClass =
        inference::localModelArgmax(
            probabilities
        );

    if (
        predictedClass < 0
        || predictedClass
            >= static_cast<int>(
                inference::LOCAL_CLASS_COUNT
            )
    ) {
        fatal(
            "invalid predicted class."
        );
    }


    ++windowCount;


    const uint32_t featureUs =
        featureEndUs - featureStartUs;

    const uint32_t normalizeUs =
        normalizeEndUs - featureEndUs;

    const uint32_t invokeUs =
        invokeEndUs - normalizeEndUs;

    const uint32_t pipelineUs =
        invokeEndUs - pipelineStartUs;


    float meanPeriodMs = 0.0f;
    float minPeriodMs = 0.0f;
    float maxPeriodMs = 0.0f;

    if (periodCount > 0) {
        meanPeriodMs =
            static_cast<float>(
                periodSumUs
            )
            / static_cast<float>(
                periodCount
            )
            / 1000.0f;

        minPeriodMs =
            static_cast<float>(
                minPeriodUs
            )
            / 1000.0f;

        maxPeriodMs =
            static_cast<float>(
                maxPeriodUs
            )
            / 1000.0f;
    }


    float stepMs = 0.0f;

    if (previousWindowReadyUs != 0) {
        stepMs =
            static_cast<float>(
                windowReadyUs
                - previousWindowReadyUs
            )
            / 1000.0f;
    }

    previousWindowReadyUs =
        windowReadyUs;


    // Keep runtime logging intentionally compact so Serial output
    // is less likely to disturb 100 Hz acquisition.
    Serial.printf(
        "W=%lu S=%lu "
        "gesture=%s class=%d conf=%.6f "
        "step_ms=%.3f "
        "period_ms(mean/min/max)=%.3f/%.3f/%.3f "
        "pipe_us=%lu feat_us=%lu norm_us=%lu invoke_us=%lu "
        "overruns=%lu read_fail=%lu\n",
        static_cast<unsigned long>(
            windowCount
        ),
        static_cast<unsigned long>(
            successfulSamples
        ),
        inference::localClassName(
            static_cast<size_t>(
                predictedClass
            )
        ),
        predictedClass,
        probabilities[predictedClass],
        stepMs,
        meanPeriodMs,
        minPeriodMs,
        maxPeriodMs,
        static_cast<unsigned long>(
            pipelineUs
        ),
        static_cast<unsigned long>(
            featureUs
        ),
        static_cast<unsigned long>(
            normalizeUs
        ),
        static_cast<unsigned long>(
            invokeUs
        ),
        static_cast<unsigned long>(
            fullPeriodOverruns
        ),
        static_cast<unsigned long>(
            readFailures
        )
    );


    resetPeriodStats();
}


bool sampleOnce() {
    ImuSample sample;

    if (!mpu6050Read(sample)) {
        ++readFailures;
        return false;
    }


    const float sampleVector[
        sampling::SENSOR_CHANNELS
    ] = {
        sample.ax,
        sample.ay,
        sample.az,
        sample.gx,
        sample.gy,
        sample.gz,
    };


    const uint32_t sampleTimestampUs =
        micros();

    updatePeriodStats(
        sampleTimestampUs
    );

    ++successfulSamples;


    const bool windowReady =
        runtimeBuffer.pushSample(
            sampleVector
        );

    if (windowReady) {
        runWindowInference(
            sampleTimestampUs
        );
    }

    return true;
}


void initializeSensor() {
    if (
        !Wire.begin(
            SDA_PIN,
            SCL_PIN,
            I2C_FREQUENCY_HZ
        )
    ) {
        fatal(
            "I2C initialization failed."
        );
    }


    const uint8_t whoAmI =
        mpu6050WhoAmI();

    Serial.printf(
        "WHO_AM_I: 0x%02X\n",
        whoAmI
    );

    if (whoAmI == 0x68) {
        Serial.println(
            "Sensor identity: standard MPU6050"
        );
    } else if (whoAmI == 0x74) {
        Serial.println(
            "WARNING: non-standard WHO_AM_I=0x74; "
            "register compatibility previously verified."
        );
    } else {
        fatal(
            "unsupported sensor identity."
        );
    }


    if (!mpu6050Begin()) {
        fatal(
            "sensor configuration failed."
        );
    }


    Serial.println(
        "Sensor configuration OK."
    );

    Serial.println(
        "I2C frequency: 400 kHz"
    );

    Serial.println(
        "Sampling rate: 100 Hz"
    );

    Serial.println(
        "Accelerometer range: +/-4 g"
    );

    Serial.println(
        "Gyroscope range: +/-500 dps"
    );
}


void calibrateGyroscope() {
    Serial.println();
    Serial.println(
        "Gyro calibration starting..."
    );

    Serial.println(
        "KEEP THE DEVICE COMPLETELY STILL."
    );


    GyroBias bias;

    if (
        !mpu6050CalibrateGyro(
            bias,
            GYRO_CALIBRATION_SAMPLES
        )
    ) {
        fatal(
            "gyro calibration failed."
        );
    }


    Serial.printf(
        "Gyro bias: "
        "gx=%.4f, gy=%.4f, gz=%.4f dps\n",
        bias.x,
        bias.y,
        bias.z
    );

    Serial.println(
        "Gyro calibration complete."
    );
}


void initializeModel() {
    Serial.println();
    Serial.println(
        "Initializing production Float32 model..."
    );

    if (!inference::initLocalModel()) {
        fatal(
            "production Float32 model initialization failed."
        );
    }

    Serial.println(
        "Production Float32 model ready."
    );
}


}  // namespace


void setup() {
    Serial.begin(115200);
    delay(1500);


    Serial.println();
    Serial.println(
        "=== Phase 4 / M5 — Continuous Local TinyML Runtime ==="
    );

    Serial.printf(
        "Firmware version: %s\n",
        FIRMWARE_VERSION
    );

    Serial.printf(
        "Dataset version: %s\n",
        DATASET_VERSION
    );

    Serial.printf(
        "Accelerometer calibration: %s\n",
        ACCEL_CALIBRATION_VERSION
    );

    Serial.printf(
        "Orientation protocol: %s\n",
        ORIENTATION_VERSION
    );

    Serial.printf(
        "Runtime contract: %lu Hz, "
        "%u-sample window, "
        "%u-sample step, 50%% overlap\n",
        static_cast<unsigned long>(
            SAMPLE_RATE_HZ
        ),
        static_cast<unsigned>(
            sampling::WINDOW_SAMPLES
        ),
        static_cast<unsigned>(
            sampling::STEP_SAMPLES
        )
    );


    initializeSensor();
    calibrateGyroscope();
    initializeModel();


    runtimeBuffer.reset();
    resetPeriodStats();

    successfulSamples = 0;
    readFailures = 0;
    fullPeriodOverruns = 0;
    windowCount = 0;

    previousSampleUs = 0;
    previousWindowReadyUs = 0;


    Serial.println();
    Serial.println(
        "Continuous runtime starting."
    );

    Serial.println(
        "First prediction after 100 successful samples;"
    );

    Serial.println(
        "then one prediction every 50 new successful samples."
    );

    Serial.println(
        "For the first validation run, keep the device IDLE "
        "for at least 10 prediction windows."
    );

    Serial.println();


    nextSampleUs =
        micros() + SAMPLE_PERIOD_US;
}


void loop() {
    const uint32_t nowUs =
        micros();

    if (
        static_cast<int32_t>(
            nowUs - nextSampleUs
        ) < 0
    ) {
        delayMicroseconds(100);
        return;
    }


    const uint32_t latenessUs =
        nowUs - nextSampleUs;

    if (
        latenessUs
        >= SAMPLE_PERIOD_US
    ) {
        ++fullPeriodOverruns;

        // Do not burst-read stale "catch-up" samples.
        // Re-anchor after a full-period miss.
        nextSampleUs =
            nowUs + SAMPLE_PERIOD_US;
    } else {
        // Preserve the 100 Hz schedule while runtime work fits
        // inside the available 10 ms period.
        nextSampleUs += SAMPLE_PERIOD_US;
    }


    sampleOnce();
}

#endif
