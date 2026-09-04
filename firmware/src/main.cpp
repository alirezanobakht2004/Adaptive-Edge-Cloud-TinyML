#include <Arduino.h>
#include <Wire.h>

constexpr uint8_t SDA_PIN  = 8;
constexpr uint8_t SCL_PIN  = 9;
constexpr uint8_t MPU_ADDR = 0x68;

constexpr uint8_t REG_CONFIG       = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG  = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_PWR_MGMT_1   = 0x6B;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;

constexpr float ACCEL_SCALE = 8192.0f;  // ±4g
constexpr float GYRO_SCALE  = 65.5f;    // ±500 deg/s

bool writeReg(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
}

bool readSensor(
    int16_t &ax, int16_t &ay, int16_t &az,
    int16_t &gx, int16_t &gy, int16_t &gz
) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(REG_ACCEL_XOUT_H);

    if (Wire.endTransmission(false) != 0)
        return false;

    // accel(6) + temp(2) + gyro(6) = 14 bytes
    if (Wire.requestFrom(MPU_ADDR, (uint8_t)14) != 14)
        return false;

    ax = (int16_t)((Wire.read() << 8) | Wire.read());
    ay = (int16_t)((Wire.read() << 8) | Wire.read());
    az = (int16_t)((Wire.read() << 8) | Wire.read());

    // temperature فعلاً لازم نداریم
    Wire.read();
    Wire.read();

    gx = (int16_t)((Wire.read() << 8) | Wire.read());
    gy = (int16_t)((Wire.read() << 8) | Wire.read());
    gz = (int16_t)((Wire.read() << 8) | Wire.read());

    return true;
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println();
    Serial.println("=== Phase 1 / M2 - Stable IMU Stream ===");

    // حالا به مقدار اصلی معماری برمی‌گردیم
    Wire.begin(SDA_PIN, SCL_PIN, 400000);

    bool ok = true;

    ok &= writeReg(REG_PWR_MGMT_1, 0x00);
    delay(100);

    ok &= writeReg(REG_CONFIG, 0x04);        // DLPF
    ok &= writeReg(REG_GYRO_CONFIG, 0x08);   // +/-500 dps
    ok &= writeReg(REG_ACCEL_CONFIG, 0x08);  // +/-4 g

    Serial.println(ok ? "Configuration OK" : "Configuration FAILED");
}

void loop() {
    int16_t axRaw, ayRaw, azRaw;
    int16_t gxRaw, gyRaw, gzRaw;

    if (!readSensor(axRaw, ayRaw, azRaw, gxRaw, gyRaw, gzRaw)) {
        Serial.println("READ FAILED");
        delay(100);
        return;
    }

    float ax = axRaw / ACCEL_SCALE;
    float ay = ayRaw / ACCEL_SCALE;
    float az = azRaw / ACCEL_SCALE;

    float gx = gxRaw / GYRO_SCALE;
    float gy = gyRaw / GYRO_SCALE;
    float gz = gzRaw / GYRO_SCALE;

    Serial.printf(
        "ax=%+.3f g ay=%+.3f g az=%+.3f g | "
        "gx=%+.2f dps gy=%+.2f dps gz=%+.2f dps\n",
        ax, ay, az, gx, gy, gz
    );

    delay(100);
}