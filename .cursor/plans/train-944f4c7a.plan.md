<!-- 944f4c7a-61c3-43e0-a73c-8676fa95957a b9f2f523-ec87-4564-ba9a-d18a2f9bd64c -->
# Minimal YOLO11n Pothole Training Script

## Files to add

- `scripts/train_pothole_yolo11n.py`

## What the script will do

- Download `jocelyndumlao/multi-weather-pothole-detection-mwpd` using KaggleHub.
- Resolve dataset layout; if COCO-style, auto-convert annotations to YOLO txt for a single class `pothole`. If already YOLO, reuse.
- Auto-create a `data.yaml` for the DETECTION task pointing to `images/train` and `images/val` with `names: ['pothole']`.
- If split is missing, create an 80/20 train/val split deterministically.
- Train DETECTION with `YOLO('yolo11n.pt')` for 150 epochs at `imgsz=320`, single-class pothole, basic stdout logging, no flags.
- Save only the best weights (`best.pt`) and print its path.
- Notes: DETECTION-only; follows YOLO11 usage patterns per docs [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11).

## Key snippet (concise, non-executable here)

```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(data=data_yaml, epochs=150, imgsz=320, save=True, save_period=-1)
```

## Outputs

- Best model at `runs/detect/train*/weights/best.pt` (path printed on completion).
- Simple prints for progress and dataset prep.

## Constraints honored

- 320px input (aligned with `laica/potholed/potholed.py`).
- Basic logging only.
- No flags/CLI; a simple Python script.
- No ONNX export now; future TinyGrad pkl export will be a separate step.

### To-dos

- [ ] Download MWPD with KaggleHub and locate images/annotations
- [ ] Convert COCO to YOLO single-class labels and build data.yaml
- [ ] Train YOLO11n for 150 epochs at 320px, save only best.pt
- [ ] Print path to best.pt and basic stats