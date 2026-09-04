#include <Arduino.h>
#include <Wire.h>

#include "version.h"
#include "sensors/mpu6050.h"

constexpr uint8_t SDA_PIN = 8;
constexpr uint8_t SCL_PIN = 9;

constexpr uint32_t I2C_FREQUENCY_HZ = 400000;

constexpr uint32_t SAMPLE_RATE_HZ = 100;
constexpr uint32_t SAMPLE_PERIOD_US =
    1000000UL / SAMPLE_RATE_HZ;

constexpr uint16_t GYRO_CALIBRATION_SAMPLES = 200;

uint32_t nextSampleUs = 0;
uint32_t sampleCount = 0;
uint32_t readFailures = 0;

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println();
    Serial.println(
        "=== Phase 2 / Boot Gyroscope Calibration ==="
    );
    Serial.printf("Firmware version: %s\n", FIRMWARE_VERSION);
    Serial.printf(
        "Accelerometer calibration: %s\n",
        ACCEL_CALIBRATION_VERSION
    );
    Serial.printf("Orientation protocol: %s\n", ORIENTATION_VERSION);
    Serial.printf("Dataset target: %s\n", DATASET_VERSION);

    if (!Wire.begin(
            SDA_PIN,
            SCL_PIN,
            I2C_FREQUENCY_HZ
        )) {
        Serial.println(
            "FATAL: I2C initialization failed."
        );

        while (true) {
            delay(1000);
        }
    }

    const uint8_t whoAmI = mpu6050WhoAmI();

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
        Serial.println(
            "FATAL: unsupported sensor identity."
        );

        while (true) {
            delay(1000);
        }
    }

    if (!mpu6050Begin()) {
        Serial.println(
            "FATAL: sensor configuration failed."
        );

        while (true) {
            delay(1000);
        }
    }

    Serial.println("Sensor configuration OK.");
    Serial.println("I2C frequency: 400 kHz");
    Serial.println("Accelerometer range: +/-4 g");
    Serial.println("Gyroscope range: +/-500 dps");

    Serial.println();
    Serial.println("Gyro calibration starting...");
    Serial.println(
        "KEEP THE DEVICE COMPLETELY STILL."
    );

    GyroBias bias;

    if (!mpu6050CalibrateGyro(
            bias,
            GYRO_CALIBRATION_SAMPLES
        )) {
        Serial.println(
            "FATAL: gyro calibration failed."
        );

        while (true) {
            delay(1000);
        }
    }

    Serial.printf(
        "Gyro bias: gx=%.4f, gy=%.4f, gz=%.4f dps\n",
        bias.x,
        bias.y,
        bias.z
    );

    Serial.println(
        "Gyro calibration complete."
    );

    Serial.println();
    Serial.println("Sampling rate: 100 Hz");
    Serial.println(
        "timestamp_ms,ax,ay,az,gx,gy,gz"
    );

    nextSampleUs =
        micros() + SAMPLE_PERIOD_US;
}

void loop() {
    const uint32_t nowUs = micros();

    if (
        static_cast<int32_t>(
            nowUs - nextSampleUs
        ) < 0
    ) {
        return;
    }

    nextSampleUs += SAMPLE_PERIOD_US;

    ImuSample sample;

    if (!mpu6050Read(sample)) {
        readFailures++;
        return;
    }

    const uint32_t timestampMs = millis();

    Serial.printf(
        "%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        static_cast<unsigned long>(timestampMs),
        sample.ax,
        sample.ay,
        sample.az,
        sample.gx,
        sample.gy,
        sample.gz
    );

    sampleCount++;
}