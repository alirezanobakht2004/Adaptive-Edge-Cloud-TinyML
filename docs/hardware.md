# Hardware Baseline

## Current verified hardware

- ESP32-S3-WROOM-1-N8R2 development board
- GY-521 module with an MPU6050-compatible register interface
- Current IMU identity: non-standard `WHO_AM_I=0x74`
- Register-level compatibility required by this project has been verified on the current unit
- Sensor power: 3.3 V
- I2C frequency: 400 kHz
- SDA: GPIO8
- SCL: GPIO9
- Accelerometer range: +/-4 g
- Gyroscope range: +/-500 dps
- Sampling rate: 100 Hz

The project must not describe the current sensor as a confirmed standard MPU6050 device.
Measured hardware behavior takes precedence over generic module labeling.

## Current wiring

```text
GY-521              ESP32-S3
--------------------------------
VCC       --------> 3.3V
GND       --------> GND
SDA       --------> GPIO8
SCL       --------> GPIO9
```

GPIO8/GPIO9 have been verified on the physical development board and are no longer
provisional project pins.

## Boot gyroscope calibration

At each boot the device must remain completely still while 200 gyroscope samples are
collected at 100 Hz. The resulting bias is subtracted from subsequent gyroscope readings.

## Accelerometer calibration

Current frozen calibration version for dataset-v1:

```text
accel-cal-v1
```

The firmware applies:

```text
corrected_g = (measured_g - offset_g) / axis_scale
```

with:

```text
offset_x = 0.014300 g
scale_x  = 0.995437

offset_y = 0.019846 g
scale_y  = 1.000465

offset_z = 0.114024 g
scale_z  = 1.009354
```

A six-position post-correction validation on 2026-09-04 produced the following active-axis
means:

```text
X+  +1.001801 g
X-  -1.008400 g
Y+  +1.000596 g
Y-  -0.998428 g
Z+  +0.999007 g
Z-  -1.007722 g
```

These measurements validate the currently applied calibration for dataset-v1. They are
not a new set of raw-sensor calibration coefficients.
