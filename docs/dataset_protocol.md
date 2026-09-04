# Dataset Protocol v1

## Project

Adaptive Edge-Cloud TinyML for Gesture Recognition

## Dataset Version

dataset-v1

## Required Version Tags

- Firmware: `0.1.0`
- Accelerometer calibration: `accel-cal-v1`
- Orientation protocol: `orientation-v1`
- Feature version used later in preprocessing: `features-v1`

These identifiers must remain fixed while collecting dataset-v1. Any intentional change
must be documented before additional data are collected.

## Sensor Configuration

- IMU: GY-521 with MPU6050-compatible register interface
- Current hardware identity: non-standard `WHO_AM_I=0x74`
- Sampling rate: 100 Hz
- Sample period: 10 ms
- Accelerometer range: +/-4 g
- Gyroscope range: +/-500 dps
- I2C frequency: 400 kHz
- SDA: GPIO8
- SCL: GPIO9
- Gyroscope calibration: boot-time bias estimation using 200 samples at 100 Hz
- Accelerometer calibration: `accel-cal-v1`, applied in firmware

## Raw Sample Format

```text
timestamp_ms,ax,ay,az,gx,gy,gz
```

Units:

- Accelerometer: g, after `accel-cal-v1`
- Gyroscope: degrees per second, after boot bias subtraction
- Timestamp: milliseconds

## Gesture Classes

| ID | Class | Description |
|---:|---|---|
| 0 | IDLE | Device held still with no intended gesture |
| 1 | SWIPE_LEFT | Clear leftward movement |
| 2 | SWIPE_RIGHT | Clear rightward movement |
| 3 | ROTATE_CW | Clockwise rotation |
| 4 | SHAKE | Repeated short back-and-forth movement |

No additional gesture classes are allowed in dataset-v1.

## Physical Assembly

The ESP32-S3 board and GY-521 are mounted on the same breadboard.

The entire breadboard assembly must be moved as one rigid body during data collection.

The GY-521 must not move relative to the ESP32-S3 or breadboard.

The USB cable should remain loose enough to avoid applying significant force to the assembly.

## Home Orientation — orientation-v1

For all dataset-v1 recording sessions:

- Component side faces upward.
- Breadboard is approximately horizontal at the start of each gesture.
- ESP32 / USB end points toward the user.
- GY-521 end points away from the user.
- The same grip and physical orientation must be used for every session.

This convention is frozen as `orientation-v1` for dataset-v1.

## Measured Axis Sign Reference

The axis signs were verified experimentally using static gravity measurements. The
six-position reference used for validation is:

```text
+Z / Home Pose : az ≈ +1 g
-Z             : az ≈ -1 g
+X             : ax ≈ +1 g
-X             : ax ≈ -1 g
+Y             : ay ≈ +1 g
-Y             : ay ≈ -1 g
```

The measured sign is authoritative if a drawing or informal label conflicts with the
actual sensor stream.

## Gesture Direction Convention

Gesture directions are defined from the user's point of view while the device is in the
Home Orientation.

### IDLE

Hold the complete assembly still in the Home Orientation.

### SWIPE_LEFT

Move the entire assembly clearly toward the user's left.

### SWIPE_RIGHT

Move the entire assembly clearly toward the user's right.

### ROTATE_CW

Rotate the entire assembly clockwise as viewed from above.

### SHAKE

Perform repeated short back-and-forth movements while maintaining approximately the same
grip and orientation.

## Collection Rules

- Perform boot gyroscope calibration while the device is completely still.
- Do not begin a gesture during calibration.
- Use the same grip and `orientation-v1` across dataset-v1 sessions.
- Move the full breadboard assembly, not the sensor independently.
- Avoid pulling on the USB cable or jumper wires.
- Do not manually edit raw sensor samples.
- Preserve original timestamps.
- Store raw recordings separately from processed/windowed data.
- Keep recording sessions separate to support session-based train/validation/test splits.
- Record firmware, calibration, orientation, user, session, and gesture metadata.

## Windowing Configuration

Initial model window:

```text
1.0 second
100 samples
6 IMU channels
```

Runtime step:

```text
0.5 second
50% overlap
```

Windowing is performed after raw data collection.

## Versioning

Any change to the following requires explicit documentation and may require a new dataset
version:

- Sampling rate
- Sensor range
- Physical orientation
- Axis interpretation
- Gesture definitions
- Raw data schema
- Calibration procedure or calibration version
- Firmware behavior affecting recorded samples
