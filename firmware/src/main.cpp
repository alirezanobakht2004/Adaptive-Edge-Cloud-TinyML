#ifndef PIO_UNIT_TESTING

#include <Arduino.h>
#include <Wire.h>

#include <climits>

#include "version.h"
#include "sensors/mpu6050.h"
#include "feature_extractor.h"
#include "input_preprocessor.h"
#include "prefix_runner.h"
#include "uncertainty.h"
#include "window_buffer.h"


namespace {

constexpr uint8_t SDA_PIN = 8;
constexpr uint8_t SCL_PIN = 9;

constexpr uint32_t I2C_FREQUENCY_HZ = 400000;

constexpr uint32_t SAMPLE_RATE_HZ = 100;
constexpr uint32_t SAMPLE_PERIOD_US =
    1000000UL / SAMPLE_RATE_HZ;

constexpr uint16_t GYRO_CALIBRATION_SAMPLES = 200;

constexpr const char* FEATURE_VERSION =
    "features-v1";

constexpr const char* UNCERTAINTY_MODEL_VERSION =
    "gesture-model-v1.1.0";


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
    "Feature count must match normalization input."
);

static_assert(
    inference::MODEL_INPUT_FEATURES
        == inference::PREFIX_INPUT_FEATURES,
    "Normalization output must match prefix input."
);

static_assert(
    inference::PREFIX_OUTPUT_UNITS
        == inference::EDGE_HEAD_INPUT_UNITS,
    "B3 output must match Edge Head input."
);

static_assert(
    inference::UNCERTAINTY_PASS_COUNT == 5,
    "Phase-5 runtime requires exactly five stochastic passes."
);


sampling::WindowBuffer runtimeBuffer;

float inferenceWindow[
    sampling::WINDOW_SAMPLES
][
    sampling::SENSOR_CHANNELS
];


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


const char* gestureClassName(
    size_t classIndex
) {
    static const char* const names[
        inference::EDGE_HEAD_CLASS_COUNT
    ] = {
        "IDLE",
        "SWIPE_LEFT",
        "SWIPE_RIGHT",
        "ROTATE_CW",
        "SHAKE",
    };

    if (
        classIndex
        >= inference::EDGE_HEAD_CLASS_COUNT
    ) {
        return "UNKNOWN";
    }

    return names[classIndex];
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

    previousSampleUs =
        sampleTimestampUs;
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

    float embedding[
        inference::PREFIX_OUTPUT_UNITS
    ] = {};

    inference::UncertaintyResult
        uncertaintyResult = {};

    inference::UncertaintyRuntimeDiagnostics
        uncertaintyDiagnostics = {};


    // Same timing convention used in Phase 4:
    // window-buffer copy is intentionally outside the measured
    // compute pipeline.
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
        !inference::runPrefixB3(
            normalized,
            embedding
        )
    ) {
        fatal(
            "B1/B2/B3 Float32 prefix inference failed."
        );
    }

    const uint32_t prefixEndUs =
        micros();


    if (
        !inference::
            runStochasticUncertaintyFromEmbedding(
                embedding,
                uncertaintyResult,
                &uncertaintyDiagnostics
            )
    ) {
        fatal(
            "five-pass uncertainty inference failed."
        );
    }

    const uint32_t uncertaintyEndUs =
        micros();


    const int predictedClass =
        uncertaintyResult.predictedClass;

    if (
        predictedClass < 0
        || predictedClass
            >= static_cast<int>(
                inference::EDGE_HEAD_CLASS_COUNT
            )
    ) {
        fatal(
            "invalid predicted class."
        );
    }


    ++windowCount;


    const uint32_t featureUs =
        featureEndUs
        - featureStartUs;

    const uint32_t normalizeUs =
        normalizeEndUs
        - featureEndUs;

    const uint32_t prefixUs =
        prefixEndUs
        - normalizeEndUs;

    const uint32_t uncertaintyUs =
        uncertaintyEndUs
        - prefixEndUs;

    const uint32_t pipelineUs =
        uncertaintyEndUs
        - pipelineStartUs;


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


    // Keep one compact line per 0.5 s inference step.
    //
    // "unc" is normalized predictive entropy in [0,1].
    // "var" is mean population variance across the 5 class
    // probabilities and 5 stochastic passes.
    // "masks" and "prange" are diagnostics only; they do not
    // participate in any LOCAL/OFFLOAD decision.
    Serial.printf(
        "W=%lu S=%lu "
        "g=%s c=%d conf=%.6f unc=%.6f var=%.7f "
        "masks=%u prange=%.6f "
        "step_ms=%.3f per_ms=%.3f/%.3f/%.3f "
        "pipe_us=%lu feat=%lu norm=%lu prefix=%lu mc=%lu "
        "ov=%lu rf=%lu\n",
        static_cast<unsigned long>(
            windowCount
        ),
        static_cast<unsigned long>(
            successfulSamples
        ),
        gestureClassName(
            static_cast<size_t>(
                predictedClass
            )
        ),
        predictedClass,
        uncertaintyResult.confidence,
        uncertaintyResult.
            normalizedPredictiveEntropy,
        uncertaintyResult.meanClassVariance,
        static_cast<unsigned>(
            uncertaintyDiagnostics.
                uniqueMaskCount
        ),
        uncertaintyDiagnostics.
            maxPassProbabilityRange,
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
            prefixUs
        ),
        static_cast<unsigned long>(
            uncertaintyUs
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


void initializeUncertaintyModel() {
    Serial.println();

    Serial.println(
        "Initializing Phase-5 B1/B2/B3 Float32 prefix..."
    );

    if (!inference::initPrefixRunner()) {
        fatal(
            "Phase-5 prefix initialization failed."
        );
    }

    Serial.printf(
        "Prefix tensor arena: %u / %u bytes\n",
        static_cast<unsigned>(
            inference::
                prefixRunnerTensorArenaUsedBytes()
        ),
        static_cast<unsigned>(
            inference::
                prefixRunnerTensorArenaCapacityBytes()
        )
    );


    const uint32_t prngSeed =
        inference::
            seedUncertaintyMaskPrngFromDevice();

    Serial.printf(
        "MC-Dropout PRNG seed: 0x%08lX\n",
        static_cast<unsigned long>(
            prngSeed
        )
    );

    Serial.println(
        "Mask generator: xorshift32; "
        "device-derived seed; true-randomness claim: NO"
    );

    Serial.println(
        "Phase-5 uncertainty inference ready."
    );
}


}  // namespace


void setup() {
    Serial.begin(115200);
    delay(1500);


    Serial.println();

    Serial.println(
        "=== Phase 5 / M6 — Continuous On-Device Uncertainty Runtime ==="
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
        "Feature version: %s\n",
        FEATURE_VERSION
    );

    Serial.printf(
        "Model version: %s\n",
        UNCERTAINTY_MODEL_VERSION
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

    Serial.printf(
        "Uncertainty contract: %u stochastic passes; "
        "score=normalized predictive entropy\n",
        static_cast<unsigned>(
            inference::
                UNCERTAINTY_PASS_COUNT
        )
    );

    Serial.println(
        "Offload threshold/policy: NONE"
    );


    initializeSensor();
    calibrateGyroscope();
    initializeUncertaintyModel();


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
        "Continuous uncertainty runtime starting."
    );

    Serial.println(
        "First inference after 100 successful samples;"
    );

    Serial.println(
        "then one inference every 50 new successful samples."
    );

    Serial.println(
        "For the stability run, keep the device IDLE "
        "for at least 20 inference windows."
    );

    Serial.println();


    nextSampleUs =
        micros()
        + SAMPLE_PERIOD_US;
}


void loop() {
    const uint32_t nowUs =
        micros();

    if (
        static_cast<int32_t>(
            nowUs
            - nextSampleUs
        ) < 0
    ) {
        delayMicroseconds(100);
        return;
    }


    const uint32_t latenessUs =
        nowUs
        - nextSampleUs;

    if (
        latenessUs
        >= SAMPLE_PERIOD_US
    ) {
        ++fullPeriodOverruns;

        // Do not burst-read stale "catch-up" samples.
        // Re-anchor after a full-period miss.
        nextSampleUs =
            nowUs
            + SAMPLE_PERIOD_US;
    } else {
        // Preserve the 100 Hz schedule while the complete
        // Phase-5 uncertainty pipeline fits inside the 10 ms period.
        nextSampleUs +=
            SAMPLE_PERIOD_US;
    }


    sampleOnce();
}

#endif
