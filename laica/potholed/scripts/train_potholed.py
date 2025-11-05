#!/usr/bin/env python3
"""
Minimal YOLO11s Pothole Detection Training Script
Downloads MWPD dataset via KaggleHub and trains YOLO11s for pothole detection

Single-Stage Training:
  Trains at 512x512 resolution (matches inference resolution and cropped input)

Resume Training:
  The script automatically detects and resumes from checkpoints if training is interrupted.
  Simply re-run the script to resume from the last checkpoint.

Models are exported to:
  - laica/potholed/models/model_fp16.onnx (FP16 ONNX model)
"""
import os
import sys
import json
import shutil
import kagglehub
from pathlib import Path
from typing import Dict, Any
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
RUNS_DIR = SCRIPT_DIR / "runs"
DATA_DIR = RUNS_DIR / "data"

def download_dataset() -> Path:
    """Download MWPD dataset using KaggleHub"""

    print("=" * 80)
    print("STEP 1: Downloading MWPD dataset from Kaggle...")
    print("=" * 80)

    # Download dataset
    dataset_path = kagglehub.dataset_download("jocelyndumlao/multi-weather-pothole-detection-mwpd")
    dataset_path = Path(dataset_path)

    print(f"✅ Dataset downloaded to: {dataset_path}")

    # List contents
    print("\nDataset structure:")
    for item in sorted(dataset_path.rglob("*"))[:20]:  # Show first 20 items
        if item.is_file():
            print(f"  {item.relative_to(dataset_path)}")

    return dataset_path


def find_dataset_structure(dataset_path: Path) -> Dict[str, Path]:
    """Find images and labels in the dataset - checks multiple levels deep"""

    print("\n" + "=" * 80)
    print("STEP 2: Analyzing dataset structure...")
    print("=" * 80)

    paths = {}
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Detected subdirs in dataset: {subdirs}")

    splits = ['train', 'valid', 'test']

    # Level 1: Check direct subdirs (train/, valid/, test/)
    for split in splits:
        key_split = 'val' if split == 'valid' else split
        if split in subdirs:
            split_dir = dataset_path / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            if images_dir.exists():
                paths[f'{key_split}_images'] = images_dir
                print(f"  Found {key_split} images: {images_dir}")
            if labels_dir.exists():
                paths[f'{key_split}_labels'] = labels_dir
                print(f"  Found {key_split} labels: {labels_dir}")

    # Level 2: Check one level deeper
    if not paths:
        for subdir in subdirs:
            sub_path = dataset_path / subdir
            if sub_path.is_dir():
                inner_subdirs = [d for d in os.listdir(sub_path) if os.path.isdir(sub_path / d)]
                print(f"Checking inner subdirs in {subdir}: {inner_subdirs}")
                for split in splits:
                    key_split = 'val' if split == 'valid' else split
                    if split in inner_subdirs:
                        split_dir = sub_path / split
                        images_dir = split_dir / 'images'
                        labels_dir = split_dir / 'labels'
                        if images_dir.exists():
                            paths[f'{key_split}_images'] = images_dir
                            print(f"  Found {key_split} images: {images_dir}")
                        if labels_dir.exists():
                            paths[f'{key_split}_labels'] = labels_dir
                            print(f"  Found {key_split} labels: {labels_dir}")
                if paths:
                    break

    # Level 3: Check two levels deeper
    if not paths:
        for subdir in subdirs:
            sub_path = dataset_path / subdir
            if sub_path.is_dir():
                inner_subdirs = [d for d in os.listdir(sub_path) if os.path.isdir(sub_path / d)]
                for inner_subdir in inner_subdirs:
                    inner_path = sub_path / inner_subdir
                    if inner_path.is_dir():
                        deepest_subdirs = [d for d in os.listdir(inner_path) if os.path.isdir(inner_path / d)]
                        print(f"Checking deepest subdirs in {inner_subdir}: {deepest_subdirs}")
                        for split in splits:
                            key_split = 'val' if split == 'valid' else split
                            if split in deepest_subdirs:
                                split_dir = inner_path / split
                                images_dir = split_dir / 'images'
                                labels_dir = split_dir / 'labels'
                                if images_dir.exists():
                                    paths[f'{key_split}_images'] = images_dir
                                    print(f"  Found {key_split} images: {images_dir}")
                                if labels_dir.exists():
                                    paths[f'{key_split}_labels'] = labels_dir
                                    print(f"  Found {key_split} labels: {labels_dir}")
                        if paths:
                            break
                if paths:
                    break

    # Determine format
    structure = {
        'images_train': paths.get('train_images'),
        'images_val': paths.get('val_images'),
        'labels_train': paths.get('train_labels'),
        'labels_val': paths.get('val_labels'),
        'format': 'yolo' if paths else None
    }

    print(f"\n✅ Dataset format detected: {structure['format']}")
    print(f"  Train images: {structure['images_train']}")
    print(f"  Val images: {structure['images_val']}")
    print(f"  Train labels: {structure['labels_train']}")
    print(f"  Val labels: {structure['labels_val']}")

    return structure


def convert_coco_to_yolo(coco_json_path: Path, images_dir: Path, output_dir: Path, class_name: str = 'pothole'):
    """Convert COCO format annotations to YOLO format (single class)"""

    print(f"\nConverting COCO annotations: {coco_json_path.name}")

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Get image dimensions
    image_info = {img['id']: (img['file_name'], img['width'], img['height'])
                  for img in coco_data['images']}

    # Process annotations
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    for img_id, (filename, img_w, img_h) in image_info.items():
        if img_id not in annotations_by_image:
            continue  # Skip images without annotations

        # Create YOLO format txt file
        txt_filename = Path(filename).stem + '.txt'
        txt_path = output_dir / txt_filename

        with open(txt_path, 'w') as f:
            for ann in annotations_by_image[img_id]:
                # COCO bbox: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']

                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width_norm = w / img_w
                height_norm = h / img_h

                # Single class (pothole = 0)
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        converted_count += 1

    print(f"  ✅ Converted {converted_count} images with annotations")
    return converted_count


def create_yolo_dataset(dataset_path: Path, structure: Dict, output_base: Path) -> Path:
    """Create YOLO-format dataset structure"""

    print("\n" + "=" * 80)
    print("STEP 3: Preparing YOLO dataset structure...")
    print("=" * 80)

    # Create output directories
    yolo_dataset = output_base / "mwpd_yolo"
    yolo_dataset.mkdir(parents=True, exist_ok=True)

    train_images = yolo_dataset / "images" / "train"
    val_images = yolo_dataset / "images" / "val"
    train_labels = yolo_dataset / "labels" / "train"
    val_labels = yolo_dataset / "labels" / "val"

    train_images.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    if structure['format'] == 'yolo':
        # Copy existing YOLO format
        print("\nCopying YOLO format dataset...")

        # Copy training set
        if structure['images_train'] and structure['labels_train']:
            print("\nProcessing training set...")
            print(f"  Source images: {structure['images_train']}")
            print(f"  Source labels: {structure['labels_train']}")

            img_count = 0
            for img_file in structure['images_train'].glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    shutil.copy2(img_file, train_images / img_file.name)
                    img_count += 1

            label_count = 0
            for label_file in structure['labels_train'].glob("*.txt"):
                shutil.copy2(label_file, train_labels / label_file.name)
                label_count += 1

            print(f"  Copied {img_count} images, {label_count} labels")

        # Copy validation set
        if structure['images_val'] and structure['labels_val']:
            print("\nProcessing validation set...")
            print(f"  Source images: {structure['images_val']}")
            print(f"  Source labels: {structure['labels_val']}")

            img_count = 0
            for img_file in structure['images_val'].glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    shutil.copy2(img_file, val_images / img_file.name)
                    img_count += 1

            label_count = 0
            for label_file in structure['labels_val'].glob("*.txt"):
                shutil.copy2(label_file, val_labels / label_file.name)
                label_count += 1

            print(f"  Copied {img_count} images, {label_count} labels")

    # Count files
    train_img_count = len(list(train_images.glob("*")))
    val_img_count = len(list(val_images.glob("*")))
    train_label_count = len(list(train_labels.glob("*.txt")))
    val_label_count = len(list(val_labels.glob("*.txt")))

    print("\n✅ Dataset prepared:")
    print(f"  Train: {train_img_count} images, {train_label_count} labels")
    print(f"  Val: {val_img_count} images, {val_label_count} labels")

    # Create data.yaml
    data_yaml_path = yolo_dataset / "data.yaml"
    with open(data_yaml_path, 'w') as f:
        f.write("# MWPD Dataset for YOLO11n Pothole Detection\n")
        f.write(f"path: {yolo_dataset.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("# Classes\n")
        f.write("names:\n")
        f.write("  0: pothole\n")

    print(f"\n✅ Created data.yaml: {data_yaml_path}")

    return data_yaml_path


def train_yolo(train_name: str, data_yaml_path: Path, yolo_model: str = 'yolo11s.pt', **train_kwargs: Any):
    """
    Train YOLO model with configurable parameters.

    Args:
        train_name: Name for the training run (used for output directory)
        data_yaml_path: Path to the dataset YAML file
        yolo_model: YOLO model to use (e.g., 'yolo11s.pt', 'yolo11n.pt', 'yolo11m.pt')
        **train_kwargs: Additional parameters passed directly to model.train()
    """
    print("\n" + "=" * 80)
    print(f"STEP 4: Training YOLO model ({train_name})...")
    print("=" * 80)
    print("\nTraining strategy:")
    print(f"  Model: {yolo_model}")
    print(f"  Training name: {train_name}")
    print("=" * 80)

    # Check for existing checkpoints to resume training
    train_dir = Path(f'runs/{train_name}')
    train_last = train_dir / 'weights' / 'last.pt'
    train_best = train_dir / 'weights' / 'best.pt'

    # Check if training can be resumed or is already complete
    if train_best.exists():
        print(f"\n✅ Training already completed! Found best model: {train_best}")
        print("  Training is complete!")
        best_model_path = train_best
        skip_training = True
    elif train_last.exists():
        print(f"\n⚠️ Found checkpoint: {train_last}")
        print("  Resuming training from last checkpoint...")
        model = YOLO(str(train_last))
        skip_training = False
        resume_training = True
    else:
        print(f"\nLoading YOLO pretrained model: {yolo_model}")
        model = YOLO(yolo_model)
        skip_training = False
        resume_training = False

    results = None
    if not skip_training:
        # Set default training parameters
        default_train_kwargs = dict(
            data=str(data_yaml_path),

            epochs=100,
            imgsz=640,
            batch=-1, # auto-batching
            lr0=0.01, # For SGD
            momentum=0.937,
            weight_decay=5e-4,
            warmup_epochs=3.0,
            patience=20,
            optimizer='SGD',

            # Learning rate scheduler
            cos_lr=False,  # Linear decay is default
            lrf=0.01, # Final LR fraction (1% of lr0)

            # Augmentations
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.0,
            degrees=0.0,
            shear=0.0,
            perspective=0.0,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,

            # YOLOv11-specific
            amp=True, # Mixed precision (FP16)
            close_mosaic=10, # Disable mosaic in last 10 epochs
            pretrained=True, # Load pretrained YOLOv11s backbone

            # Logging and saving
            resume=resume_training,
            save=True,
            save_period=-1,
            project='runs',
            name=train_name,
            exist_ok=True,
            profile=True,
            verbose=True,
        )

        # Merge user-provided kwargs with defaults (user kwargs take precedence)
        final_train_kwargs = {**default_train_kwargs, **train_kwargs}

        # Print training configuration
        print("\nTraining configuration:")
        print(f"  Model: {yolo_model}")
        print(f"  Training name: {train_name}")
        print(f"  Epochs: {final_train_kwargs.get('epochs', 'N/A')}")
        print(f"  Image size: {final_train_kwargs.get('imgsz', 'N/A')}")
        print(f"  Batch size: {final_train_kwargs.get('batch', 'N/A')}")
        print(f"  Precision: {'FP16 (mixed precision)' if final_train_kwargs.get('amp', False) else 'FP32'}")
        print(f"  Learning rate: {final_train_kwargs.get('lr0', 'N/A')}")
        print(f"  Momentum: {final_train_kwargs.get('momentum', 'N/A')}")
        print(f"  Weight decay: {final_train_kwargs.get('weight_decay', 'N/A')}")
        if final_train_kwargs.get('patience'):
            print(f"  Early stopping patience: {final_train_kwargs.get('patience')}")
        print("  Data augmentation:")
        print(f"    - Horizontal flip: {final_train_kwargs.get('fliplr', 0)}")
        print(f"    - Rotation: ±{final_train_kwargs.get('degrees', 0)}°")
        print(f"    - Shear: ±{final_train_kwargs.get('shear', 0)}°")
        print(f"    - HSV augmentation: h={final_train_kwargs.get('hsv_h', 0)}, s={final_train_kwargs.get('hsv_s', 0)}, v={final_train_kwargs.get('hsv_v', 0)}")
        print(f"    - Scale: {final_train_kwargs.get('scale', 0)}")
        print(f"    - Mosaic: {final_train_kwargs.get('mosaic', 0)}")
        print(f"    - Mixup: {final_train_kwargs.get('mixup', 0)}")
        print(f"    - Copy-paste: {final_train_kwargs.get('copy_paste', 0)}")

        if resume_training:
            print("\n▶️ RESUMING training from checkpoint...\n")
        else:
            print("\nStarting training...\n")

        # Train model with merged parameters
        results = model.train(**final_train_kwargs)

        # Get best model
        best_model_path = Path(model.trainer.save_dir) / "weights" / "best.pt"

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\n✅ Best model: {best_model_path}")

    # Print final metrics
    if results is not None and hasattr(results, 'results_dict'):
        metrics = results.results_dict
        # Get imgsz from train_kwargs or use default
        imgsz = train_kwargs.get('imgsz', 416)
        print(f"\nFinal metrics (at {imgsz}):")
        if 'metrics/precision(B)' in metrics:
            print(f"  Precision: {metrics['metrics/precision(B)']:.4f}")
        if 'metrics/recall(B)' in metrics:
            print(f"  Recall: {metrics['metrics/recall(B)']:.4f}")
        if 'metrics/mAP50(B)' in metrics:
            print(f"  mAP@50: {metrics['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"  mAP@50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
    elif skip_training:
        print("\n(Training was already complete - metrics available in training logs)")

    return best_model_path


def export_onnx(model_path: Path, imgsz: int = 512) -> Path:
    """
    Export trained model to ONNX (FP16 by default) and return the path.

    Args:
        model_path: Path to the trained .pt model
        imgsz: Image size for export (default: 512)
    """

    precision = "FP16"
    print("\n" + "=" * 80)
    print(f"Exporting model to {precision} ONNX")
    print("=" * 80)
    print(f"\nSource model: {model_path}")

    # Output to same directory as model_path
    output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    export_result = model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=True,
        dynamic=False,
        opset=12,
        half=True,  # FP16 export
        verbose=False,
        project=str(output_dir),
        name=None  # Use default naming (will create model.onnx in output_dir)
    )

    if isinstance(export_result, (str, Path)):
        onnx_path = Path(export_result)
    else:
        # Default ONNX filename is model.onnx in the output directory
        onnx_path = output_dir / 'model.onnx'

    if not onnx_path.exists():
        # Try alternative naming based on model filename
        model_stem = model_path.stem  # e.g., 'best' from 'best.pt'
        alt_path = output_dir / f'{model_stem}.onnx'
        if alt_path.exists():
            onnx_path = alt_path
        else:
            print(f"ERROR: ONNX export not found at {onnx_path}")
            sys.exit(1)

    print(f"✅ ONNX ({precision}) saved: {onnx_path}")
    return onnx_path

def main():
    print("\n" + "=" * 80)
    print("YOLO Pothole Detection Training")
    print("Dataset: Multi-Weather Pothole Detection (MWPD)")
    print("=" * 80)

    # Download dataset
    dataset_path = download_dataset()

    # Analyze dataset structure
    structure = find_dataset_structure(dataset_path)

    if structure['format'] is None:
        print("\nERROR: Could not determine dataset format")
        sys.exit(1)

    # Prepare YOLO dataset
    data_yaml_path = create_yolo_dataset(dataset_path, structure, DATA_DIR)

    # Train model
    variants = dict(
        potholed_yolo11n_640=('yolo11n.pt', dict(imgsz=640)),
        potholed_yolo11n_512=('yolo11n.pt', dict(imgsz=512)),
        potholed_yolo11n_416=('yolo11n.pt', dict(imgsz=416)),
        potholed_yolo11m_640=('yolo11m.pt', dict(imgsz=640)),
        potholed_yolo11m_512=('yolo11m.pt', dict(imgsz=512)),
        potholed_yolo11m_416=('yolo11m.pt', dict(imgsz=416)),
        potholed_yolo11s_640=('yolo11s.pt', dict(imgsz=640)),
        potholed_yolo11s_512=('yolo11s.pt', dict(imgsz=512)),
        potholed_yolo11s_416=('yolo11s.pt', dict(imgsz=416)),
    )

    for name, (model, kwargs) in variants.items():
        best_model_path = train_yolo(name, data_yaml_path, model, **kwargs)
        onnx_fp16_path = export_onnx(best_model_path, imgsz=512)
        print(f"\nFP16 ONNX model: {onnx_fp16_path}")

        # Copy the ONNX artifact into laica/potholed/models directory
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = MODELS_DIR / Path(onnx_fp16_path).name
        try:
            shutil.copy2(onnx_fp16_path, dest_path)
            print(f"\n✅ Copied model to: {dest_path}")
        except Exception as e:
            print(f"\n⚠️ Failed to copy model to {dest_path}: {e}")

    print("\n" + "=" * 80)
    print("✅ ALL DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

