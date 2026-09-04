# Dataset Protocol v1

## Project

Adaptive Edge-Cloud TinyML for Gesture Recognition

## Dataset Version

dataset-v1

## Sensor Configuration

- IMU: GY-521 / MPU6050-compatible device
- Sampling rate: 100 Hz
- Sample period: 10 ms
- Accelerometer range: +/-4 g
- Gyroscope range: +/-500 dps
- I2C frequency: 400 kHz
- SDA: GPIO8
- SCL: GPIO9
- Gyroscope calibration: boot-time bias estimation using 200 samples at 100 Hz

## Raw Sample Format

```text
timestamp_ms,ax,ay,az,gx,gy,gz
```

Units:

- Accelerometer: g
- Gyroscope: degrees per second
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

## Home Orientation

For all dataset-v1 recording sessions:

- Component side faces upward.
- Breadboard is approximately horizontal at the start of each gesture.
- ESP32 / USB end is on the user's left.
- GY-521 end is on the user's right.
- The same physical orientation must be used for every session.

## Measured Axis Mapping

The axis mapping was determined experimentally using static gravity measurements.

### Z Axis

Flat orientation with component side upward:

```text
az ≈ +1 g
```

Therefore:

```text
+Z = outward from the component surface
-Z = downward through the breadboard
```

### Y Axis

With the GY-521 end raised vertically:

```text
ay ≈ +1 g
```

Therefore:

```text
+Y = ESP32/USB end -> GY-521 end
-Y = GY-521 end -> ESP32/USB end
```

### X Axis

With the far long edge raised and the near long edge lowered:

```text
ax ≈ -1 g
```

Therefore:

```text
+X = toward the user / near long edge
-X = away from the user / far long edge
```

## Coordinate Frame Summary

```text
+X = toward user
+Y = toward GY-521
+Z = upward from component side
```

## Gesture Direction Convention

Gesture directions are defined from the user's point of view while the device is in the Home Orientation.

### IDLE

Hold the complete assembly still in the Home Orientation.

### SWIPE_LEFT

Move the entire assembly clearly toward the user's left.

### SWIPE_RIGHT

Move the entire assembly clearly toward the user's right.

### ROTATE_CW

Rotate the entire assembly clockwise as viewed from above.

### SHAKE

Perform repeated short back-and-forth movements while maintaining approximately the same grip and orientation.

## Collection Rules

- Perform boot gyroscope calibration while the device is completely still.
- Do not begin a gesture during calibration.
- Use the same grip and Home Orientation across dataset-v1 sessions.
- Move the full breadboard assembly, not the sensor independently.
- Avoid pulling on the USB cable or jumper wires.
- Do not manually edit raw sensor samples.
- Preserve original timestamps.
- Store raw recordings separately from processed/windowed data.
- Keep recording sessions separate to support session-based train/validation/test splits.

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

Any change to the following requires explicit documentation and may require a new dataset version:

- Sampling rate
- Sensor range
- Physical orientation
- Axis interpretation
- Gesture definitions
- Raw data schema
- Calibration procedure
