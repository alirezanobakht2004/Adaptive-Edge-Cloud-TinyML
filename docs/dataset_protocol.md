# Dataset Protocol — dataset-v1

## Purpose
Collect a leakage-resistant real IMU dataset for the five frozen v1 gesture classes.

## Sensor settings
- Sampling rate: 100 Hz
- Window: 1000 ms
- Samples/window: 100
- Channels: `ax, ay, az, gx, gy, gz`
- Accelerometer range: ±4 g
- Gyroscope range: ±500 °/s

## Gesture classes
1. `IDLE`
2. `SWIPE_LEFT`
3. `SWIPE_RIGHT`
4. `ROTATE_CW`
5. `SHAKE`

## Orientation
Before the first real recording session, photograph and document the fixed grip/orientation.
Do not collect `dataset-v1` until this convention is fixed.

## Sessions
Target per class:
- training session: ~120 windows
- validation session: ~40 windows
- test session: ~40 windows

Do not randomly split windows from a single continuous recording across train/test.

## Raw CSV schema
```csv
timestamp_ms,ax,ay,az,gx,gy,gz
```

Requirements:
- timestamps strictly increasing
- approximately 10 ms between samples
- exactly 100 samples for a canonical 1-second window
- no missing/non-finite values

## File layout
```text
data/raw/user_01/session_01/<gesture>_NNN.csv
```

## Metadata
Record at minimum:
- gesture
- user
- session
- sampling rate
- window length
- accel range
- gyro range
- firmware version
- notes
