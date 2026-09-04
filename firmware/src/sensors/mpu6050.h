#pragma once

#include <Arduino.h>

struct ImuSample {
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
};

struct GyroBias {
    float x;
    float y;
    float z;
};

bool mpu6050Begin();

bool mpu6050Read(ImuSample& sample);

uint8_t mpu6050WhoAmI();

bool mpu6050CalibrateGyro(
    GyroBias& bias,
    uint16_t sampleCount = 200
);

void mpu6050SetGyroBias(const GyroBias& bias);