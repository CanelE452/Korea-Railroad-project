# Geometry V2 live inference on Windows

This program combines:

- an OpenCV camera stream;
- TFmini-S height readings from the CH340 serial adapter;
- the bundled DLA-18 no-DCN geometry_v2 checkpoint;
- a live dashboard with laser, height, latency, FPS, and 2D/3D detections.

## Current hardware mapping

- External camera: OpenCV camera index `1`
- TFmini-S monitor: `COM6`, `115200` baud
- Laser line format: `L1 70cm strength=2178 ...`
- Camera-height input: `camera_height_m = L1_distance_m`

The laser's `0cm`, `1cm`, and `2cm` values are rejected as invalid. Point the
sensor at a surface inside its valid range. Inference waits whenever a valid
laser measurement is unavailable; it never substitutes a fallback height.

## Run

Double-click `run_live_windows.cmd`, or run it from Command Prompt:

```bat
cd C:\Users\DELL\Documents\GitHub\pallet-6d-pose\student_dla18_nodcn_640x480_fp32_with_inference_v4_20260723
run_live_windows.cmd
```

Use the integrated camera instead:

```bat
run_live_windows.cmd --camera-id 0
```

Write a per-frame CSV log:

```bat
run_live_windows.cmd --log-csv .\captures\live_log.csv
```

PowerShell is also supported when script execution is enabled:

```powershell
.\run_live_windows.ps1
```

## Controls

- `Esc` or `q`: quit
- `Space`: pause/resume inference
- `s`: save a dashboard image under `captures`
- `r`: clear the laser median filter

## Calibration

The default intrinsic matrix is copied from the bundled inference_v4 dataset:

```text
fx=605.9065 fy=605.9698 cx=317.5962 cy=256.2923
```

The camera frame is center-cropped to 4:3 and resized to 640x480, matching the
bundle's data preparation. If the attached camera is not the camera used to
create inference_v4, calibrate it and pass processed-image intrinsics with
`--fx`, `--fy`, `--cx`, and `--cy`.
