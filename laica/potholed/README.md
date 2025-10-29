# Pothole Detection Module

Real-time pothole detection system for openpilot using YOLOv8.

## Overview

The potholed daemon runs a YOLO-based pothole detection model on the road camera feed at 20Hz. Detections are published via the `potholeDetection` cereal message and rendered as bounding boxes on the UI.

## Running the Daemon

The daemon is automatically managed by openpilot's process manager:

```python
# In system/manager/process_config.py
PythonProcess("potholed", "selfdrive.potholed.potholed", only_onroad)
```

### Manual Testing

```bash
# Run daemon directly
python laica/potholed/potholed.py
```

## Cereal Message Structure

```capnp
struct PotholeDetection {
  frameId @0 :UInt32;
  modelExecutionTime @1 :Float32;
  potholes @2 :List(Pothole);

  struct Pothole {
    x @0 :Float32;        # Normalized x coordinate (0-1)
    y @1 :Float32;        # Normalized y coordinate (0-1)
    width @2 :Float32;    # Normalized width (0-1)
    height @3 :Float32;   # Normalized height (0-1)
    confidence @4 :Float32;  # Detection confidence (0-1)
  }
}
```

Published on service: `potholeDetection` at 20Hz

## UI Integration

Detections are rendered in the Qt UI:
- Red bounding boxes around detected potholes
- Confidence percentage label
- Only shown when `potholeDetection` message is alive

See `laica/ui/qt/onroad/annotated_camera.cc` for rendering implementation.

## References

- [YOLO Documentation](https://docs.ultralytics.com/)
- [Tinygrad](https://github.com/tinygrad/tinygrad)
