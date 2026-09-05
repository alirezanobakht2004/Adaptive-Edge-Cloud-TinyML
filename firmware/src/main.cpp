#ifndef PIO_UNIT_TESTING

#include <Arduino.h>
#include <Wire.h>

#include "version.h"
#include "sensors/mpu6050.h"
#include "feature_extractor.h"
#include "input_preprocessor.h"
#include "local_model.h"


namespace {

constexpr uint8_t SDA_PIN = 8;
constexpr uint8_t SCL_PIN = 9;

constexpr uint32_t I2C_FREQUENCY_HZ = 400000;

constexpr uint32_t SAMPLE_RATE_HZ = 100;

constexpr uint32_t SAMPLE_PERIOD_US =
    1000000UL / SAMPLE_RATE_HZ;

constexpr uint16_t GYRO_CALIBRATION_SAMPLES = 200;


static_assert(
    features::WINDOW_SAMPLES == 100,
    "Live smoke test expects a 100-sample window."
);

static_assert(
    features::SENSOR_CHANNELS == 6,
    "Live smoke test expects six IMU channels."
);

static_assert(
    features::FEATURE_COUNT
        == inference::MODEL_INPUT_FEATURES,
    "Feature count must match model input."
);


float liveWindow
    [features::WINDOW_SAMPLES]
    [features::SENSOR_CHANNELS];

uint32_t sampleTimestampsUs[
    features::WINDOW_SAMPLES
];


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


void printFloatVector(
    const char* name,
    const float* values,
    size_t count
) {
    Serial.printf("%s = [", name);

    for (size_t index = 0;
         index < count;
         ++index) {

        if (index > 0) {
            Serial.print(", ");
        }

        Serial.printf(
            "%.6f",
            values[index]
        );
    }

    Serial.println("]");
}


bool captureSingleWindow() {
    uint32_t nextSampleUs = micros();

    for (
        size_t index = 0;
        index < features::WINDOW_SAMPLES;
        ++index
    ) {
        while (
            static_cast<int32_t>(
                micros() - nextSampleUs
            ) < 0
        ) {
            delayMicroseconds(100);
        }

        sampleTimestampsUs[index] =
            micros();

        nextSampleUs += SAMPLE_PERIOD_US;

        ImuSample sample;

        if (!mpu6050Read(sample)) {
            Serial.printf(
                "IMU_READ_FAILURE at sample=%u\n",
                static_cast<unsigned>(index)
            );

            return false;
        }

        liveWindow[index][0] = sample.ax;
        liveWindow[index][1] = sample.ay;
        liveWindow[index][2] = sample.az;

        liveWindow[index][3] = sample.gx;
        liveWindow[index][4] = sample.gy;
        liveWindow[index][5] = sample.gz;
    }

    return true;
}


void printSamplingSummary() {
    uint32_t minPeriodUs = UINT32_MAX;
    uint32_t maxPeriodUs = 0;
    uint64_t periodSumUs = 0;

    for (
        size_t index = 1;
        index < features::WINDOW_SAMPLES;
        ++index
    ) {
        const uint32_t periodUs =
            sampleTimestampsUs[index]
            - sampleTimestampsUs[index - 1];

        if (periodUs < minPeriodUs) {
            minPeriodUs = periodUs;
        }

        if (periodUs > maxPeriodUs) {
            maxPeriodUs = periodUs;
        }

        periodSumUs += periodUs;
    }

    const float meanPeriodUs =
        static_cast<float>(periodSumUs)
        / static_cast<float>(
            features::WINDOW_SAMPLES - 1
        );

    const uint32_t windowSpanUs =
        sampleTimestampsUs[
            features::WINDOW_SAMPLES - 1
        ]
        - sampleTimestampsUs[0];

    Serial.println();
    Serial.println("SAMPLING SUMMARY");
    Serial.println("----------------");

    Serial.printf(
        "Samples:          %u\n",
        static_cast<unsigned>(
            features::WINDOW_SAMPLES
        )
    );

    Serial.printf(
        "Window span:      %.3f ms\n",
        static_cast<float>(windowSpanUs)
            / 1000.0f
    );

    Serial.printf(
        "Mean period:      %.3f ms\n",
        meanPeriodUs / 1000.0f
    );

    Serial.printf(
        "Min period:       %.3f ms\n",
        static_cast<float>(minPeriodUs)
            / 1000.0f
    );

    Serial.printf(
        "Max period:       %.3f ms\n",
        static_cast<float>(maxPeriodUs)
            / 1000.0f
    );
}


void runLiveInference() {
    float featureVector[
        features::FEATURE_COUNT
    ] = {};

    float normalized[
        inference::MODEL_INPUT_FEATURES
    ] = {};

    float probabilities[
        inference::LOCAL_CLASS_COUNT
    ] = {};


    if (
        !features::extractFeaturesV1(
            liveWindow,
            featureVector
        )
    ) {
        fatal("features-v1 extraction failed.");
    }


    if (
        !inference::normalizeFeaturesV1(
            featureVector,
            normalized
        )
    ) {
        fatal("feature normalization failed.");
    }


    if (
        !inference::runLocalModel(
            normalized,
            probabilities
        )
    ) {
        fatal("local Float32 inference failed.");
    }


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
        fatal("invalid predicted class.");
    }


    Serial.println();
    Serial.println("LIVE LOCAL INFERENCE");
    Serial.println("--------------------");

    printFloatVector(
        "features-v1",
        featureVector,
        features::FEATURE_COUNT
    );

    printFloatVector(
        "normalized",
        normalized,
        inference::MODEL_INPUT_FEATURES
    );

    printFloatVector(
        "probabilities",
        probabilities,
        inference::LOCAL_CLASS_COUNT
    );

    Serial.printf(
        "Predicted class:  %d\n",
        predictedClass
    );

    Serial.printf(
        "Predicted gesture: %s\n",
        inference::localClassName(
            static_cast<size_t>(
                predictedClass
            )
        )
    );

    Serial.printf(
        "Confidence:       %.6f\n",
        probabilities[predictedClass]
    );
}

}  // namespace


void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println();
    Serial.println(
        "=== Phase 4 / M5 — Live Local TinyML Smoke Test ==="
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


    if (
        !Wire.begin(
            SDA_PIN,
            SCL_PIN,
            I2C_FREQUENCY_HZ
        )
    ) {
        fatal("I2C initialization failed.");
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
        fatal("unsupported sensor identity.");
    }


    if (!mpu6050Begin()) {
        fatal("sensor configuration failed.");
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
        fatal("gyro calibration failed.");
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


    Serial.println();
    Serial.println(
        "Single-window live capture will begin."
    );

    Serial.println(
        "For the first smoke test, KEEP THE DEVICE IDLE."
    );

    Serial.println(
        "Capture starts in 3 seconds..."
    );

    delay(1000);
    Serial.println("3");

    delay(1000);
    Serial.println("2");

    delay(1000);
    Serial.println("1");

    Serial.println();
    Serial.println("CAPTURE START");


    if (!captureSingleWindow()) {
        fatal("live IMU window capture failed.");
    }


    Serial.println("CAPTURE END");

    printSamplingSummary();

    runLiveInference();


    Serial.println();
    Serial.println(
        "LIVE END-TO-END SMOKE TEST COMPLETE."
    );

    Serial.println(
        "Reset the board to run another capture."
    );
}


void loop() {
    delay(1000);
}

#endif