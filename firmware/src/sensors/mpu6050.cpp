#include "mpu6050.h"

#include <Wire.h>

namespace {

constexpr uint8_t MPU_ADDR = 0x68;

constexpr uint8_t REG_CONFIG       = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG  = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
constexpr uint8_t REG_PWR_MGMT_1   = 0x6B;
constexpr uint8_t REG_WHO_AM_I     = 0x75;

constexpr float ACCEL_SCALE = 8192.0f;  // +/-4 g
constexpr float GYRO_SCALE  = 65.5f;    // +/-500 dps

constexpr uint32_t CALIBRATION_SAMPLE_PERIOD_US = 10000UL;

GyroBias gyroBias = {
    0.0f,
    0.0f,
    0.0f
};

bool writeRegister(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.write(value);

    return Wire.endTransmission() == 0;
}

bool readRegister(uint8_t reg, uint8_t& value) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);

    if (Wire.endTransmission(false) != 0) {
        return false;
    }

    if (Wire.requestFrom(
            MPU_ADDR,
            static_cast<uint8_t>(1)
        ) != 1) {
        return false;
    }

    value = Wire.read();

    return true;
}

}  // namespace

uint8_t mpu6050WhoAmI() {
    uint8_t value = 0xFF;

    if (!readRegister(REG_WHO_AM_I, value)) {
        return 0xFF;
    }

    return value;
}

bool mpu6050Begin() {
    gyroBias = GyroBias{
        0.0f,
        0.0f,
        0.0f
    };

    if (!writeRegister(REG_PWR_MGMT_1, 0x00)) {
        return false;
    }

    delay(100);

    if (!writeRegister(REG_CONFIG, 0x04)) {
        return false;
    }

    if (!writeRegister(REG_GYRO_CONFIG, 0x08)) {
        return false;
    }

    if (!writeRegister(REG_ACCEL_CONFIG, 0x08)) {
        return false;
    }

    return true;
}

bool mpu6050Read(ImuSample& sample) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(REG_ACCEL_XOUT_H);

    if (Wire.endTransmission(false) != 0) {
        return false;
    }

    if (Wire.requestFrom(
            MPU_ADDR,
            static_cast<uint8_t>(14)
        ) != 14) {
        return false;
    }

    const int16_t axRaw =
        static_cast<int16_t>(
            (Wire.read() << 8) | Wire.read()
        );

    const int16_t ayRaw =
        static_cast<int16_t>(
            (Wire.read() << 8) | Wire.read()
        );

    const int16_t azRaw =
        static_cast<int16_t>(
            (Wire.read() << 8) | Wire.read()
        );

    Wire.read();
    Wire.read();

    const int16_t gxRaw =
        static_cast<int16_t>(
            (Wire.read() << 8) | Wire.read()
        );

    const int16_t gyRaw =
        static_cast<int16_t>(
            (Wire.read() << 8) | Wire.read()
        );

    const int16_t gzRaw =
        static_cast<int16_t>(
            (Wire.read() << 8) | Wire.read()
        );

    sample.ax =
        static_cast<float>(axRaw) / ACCEL_SCALE;

    sample.ay =
        static_cast<float>(ayRaw) / ACCEL_SCALE;

    sample.az =
        static_cast<float>(azRaw) / ACCEL_SCALE;

    sample.gx =
        (static_cast<float>(gxRaw) / GYRO_SCALE)
        - gyroBias.x;

    sample.gy =
        (static_cast<float>(gyRaw) / GYRO_SCALE)
        - gyroBias.y;

    sample.gz =
        (static_cast<float>(gzRaw) / GYRO_SCALE)
        - gyroBias.z;

    return true;
}

void mpu6050SetGyroBias(const GyroBias& bias) {
    gyroBias = bias;
}

bool mpu6050CalibrateGyro(
    GyroBias& bias,
    uint16_t sampleCount
) {
    if (sampleCount == 0) {
        return false;
    }

    const GyroBias previousBias = gyroBias;

    gyroBias = GyroBias{
        0.0f,
        0.0f,
        0.0f
    };

    double sumX = 0.0;
    double sumY = 0.0;
    double sumZ = 0.0;

    uint32_t nextSampleUs = micros();

    for (uint16_t i = 0; i < sampleCount; ++i) {
        while (
            static_cast<int32_t>(
                micros() - nextSampleUs
            ) < 0
        ) {
            delayMicroseconds(100);
        }

        nextSampleUs += CALIBRATION_SAMPLE_PERIOD_US;

        ImuSample sample;

        if (!mpu6050Read(sample)) {
            gyroBias = previousBias;
            return false;
        }

        sumX += sample.gx;
        sumY += sample.gy;
        sumZ += sample.gz;
    }

    bias.x =
        static_cast<float>(
            sumX / static_cast<double>(sampleCount)
        );

    bias.y =
        static_cast<float>(
            sumY / static_cast<double>(sampleCount)
        );

    bias.z =
        static_cast<float>(
            sumZ / static_cast<double>(sampleCount)
        );

    mpu6050SetGyroBias(bias);

    return true;
}