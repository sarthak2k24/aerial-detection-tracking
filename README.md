# Aerial Object Detection & Tracking

Real-time detection and multi-object tracking of vehicles and people in drone (UAV) video, written in C++17 with OpenCV.

A YOLOv8 model (exported to ONNX) finds objects in each frame. A tracker then gives each object a stable ID across frames, so the same car stays `#4` as it moves, even through short occlusions.

## Features

- **YOLOv8 inference** with OpenCV DNN, so no Python is needed at runtime
- **Letterboxed preprocessing** and **class-aware non-maximum suppression**
- **Multi-object tracking** with persistent IDs, track confirmation, and occlusion handling
- **Video file, camera, or stream input**, with optional annotated video output
- **Per-frame latency and FPS reporting** for benchmarking

## Architecture

```
 frame ─▶ Detector ─────────▶ nms() ─▶ Tracker ─▶ confirmed tracks ─▶ overlay / output
          (OpenCV DNN,         (Geometry)  (IoU matching,
           YOLOv8 ONNX)                    ID management)
```

| Component | Responsibility |
|---|---|
| `Detector` | Loads the ONNX model, preprocesses frames, decodes YOLOv8 output |
| `Geometry` | `Box`, `Detection`, IoU, and NMS: pure logic with no OpenCV dependency |
| `Tracker` | Matches detections to existing tracks, creates and drops tracks |
| `main.cpp` | Video I/O, drawing, timing |

Geometry and tracking logic is independent of OpenCV, so it is unit-tested in isolation.

## Build

Requires CMake 3.16+, a C++17 compiler, and OpenCV 4.x (with the `dnn` module).

```bash
# Ubuntu
sudo apt install libopencv-dev cmake g++

# Windows: install OpenCV from opencv.org and set OpenCV_DIR, or use vcpkg:
#   vcpkg install opencv4[dnn]

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Get a model

```bash
pip install ultralytics
python tools/export_model.py        # writes models/yolov8n.onnx
```

## Run

```bash
./build/detect_track models/yolov8n.onnx drone_footage.mp4
./build/detect_track models/yolov8n.onnx drone_footage.mp4 --no-display --out tracked.mp4
./build/detect_track models/yolov8n.onnx 0          # webcam
```

Press `Esc` to quit. Average latency and FPS are printed when it finishes.

## Tests

```
Iou.IdenticalBoxesIsOne            Tracker.ConfirmsTrackAfterMinHits
Iou.DisjointBoxesIsZero            Tracker.KeepsSameIdAsObjectMoves
Iou.HalfOverlap                    Tracker.SeparateObjectsGetSeparateIds
Iou.ZeroAreaBoxIsZero              Tracker.DoesNotMatchAcrossClasses
Nms.SuppressesOverlappingSameClass Tracker.SurvivesShortOcclusion
Nms.KeepsOverlappingDifferentClasses Tracker.DropsTrackAfterMaxMisses
Nms.ReturnsSortedByScore
```

## Roadmap

- [x] YOLOv8 ONNX inference with OpenCV DNN
- [x] IoU and class-aware NMS
- [x] IoU-based multi-object tracker with persistent IDs
- [x] Unit tests (GoogleTest) and CI
- [ ] Kalman filter motion prediction for each track
- [ ] Hungarian (optimal) assignment instead of greedy matching
- [ ] Benchmark script for accuracy and latency on sample flight footage
- [ ] Geolocate targets from vehicle telemetry (altitude, GPS, heading, camera FOV)
- [ ] Feed detections into [uav-ground-station](../../../uav-ground-station)

## License

MIT
