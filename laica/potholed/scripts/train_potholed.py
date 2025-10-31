#!/usr/bin/env python3
"""
Minimal YOLO11n Pothole Detection Training Script
Downloads MWPD dataset via KaggleHub and trains YOLO11n for pothole detection

Two-Stage Training:
  Stage 1: Train at 640px for rich feature learning
  Stage 2: Fine-tune at 320px for deployment optimization

Resume Training:
  The script automatically detects and resumes from checkpoints if training is interrupted.

  - If Stage 1 is interrupted: Simply re-run the script. It will resume from the last checkpoint.
  - If Stage 1 is complete: The script will skip Stage 1 and proceed to Stage 2.
  - If Stage 2 is interrupted: Re-run the script to resume Stage 2 from the last checkpoint.
  - If both stages are complete: The script will skip training and export the models.

Models are exported to:
  - laica/potholed/models/model_fp32.onnx (FP32 ONNX model)
"""
import os
import sys
import json
import shutil
import kagglehub
from pathlib import Path
from typing import Dict
from ultralytics import YOLO

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


def train_yolo11n(data_yaml_path: Path):
    """Train YOLO11n model with two-stage training approach"""
    print("\n" + "=" * 80)
    print("STEP 4: Two-Stage Training YOLO11n model...")
    print("=" * 80)
    print("\nTwo-stage training strategy:")
    print("  Stage 1: Train at 640px for rich feature learning")
    print("  Stage 2: Fine-tune at 256px for deployment optimization")
    print("=" * 80)

    # Check for existing checkpoints to resume training
    stage1_dir = Path('runs/potholed/stage1_640px')
    stage1_last = stage1_dir / 'weights' / 'last.pt'
    stage1_best = stage1_dir / 'weights' / 'best.pt'

    # =========================================================================
    # STAGE 1: High-Resolution Feature Learning (640px)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: Training at 640px for feature learning...")
    print("=" * 80)

    # Check if Stage 1 can be resumed or is already complete
    if stage1_best.exists():
        print(f"\n✅ Stage 1 already completed! Found best model: {stage1_best}")
        print("  Skipping Stage 1 training...")
        best_stage1_path = stage1_best
        skip_stage1 = True
    elif stage1_last.exists():
        print(f"\n⚠️ Found Stage 1 checkpoint: {stage1_last}")
        print("  Resuming Stage 1 training from last checkpoint...")
        model = YOLO(str(stage1_last))
        skip_stage1 = False
        resume_stage1 = True
    else:
        print("\nLoading YOLO11n pretrained model...")
        model = YOLO('yolo11n.pt')
        skip_stage1 = False
        resume_stage1 = False

    if not skip_stage1:
        print("\nStage 1 configuration:")
        print("  Model: YOLO11n")
        print("  Epochs: 100")
        print("  Image size: 640x640 (high resolution)")
        print("  Batch size: 16 (increased from 8)")
        print("  Optimizer: SGD")
        print("  Learning rate: 0.01")
        print("  Momentum: 0.937")
        print("  Weight decay: 0.0005")
        print("  Data augmentation:")
        print("    - Horizontal flip: 0.5")
        print("    - Vertical flip: 0.5")
        print("    - Rotation: ±10°")
        print("    - Shear: ±2°")
        print("    - HSV augmentation: h=0.020, s=0.8, v=0.5")
        print("    - Scale: 0.7")
        print("    - Mosaic: 1.0")
        print("    - Mixup: 0.1")

        if resume_stage1:
            print("\n⚠️ RESUMING Stage 1 training from checkpoint...\n")
        else:
            print("\nStarting Stage 1 training...\n")

        # Train Stage 1
        results_stage1 = model.train(
            data=str(data_yaml_path),
            epochs=200,
            imgsz=640,
            batch=16,
            optimizer='SGD',
            lr0=0.01,
            momentum=0.937,
            weight_decay=0.0005,

            # Data augmentation
            hsv_h=0.020,
            hsv_s=0.8,
            hsv_v=0.5,
            fliplr=0.5,
            flipud=0.5,
            degrees=10.0,
            shear=2.0,
            scale=0.7,
            mosaic=1.0,
            mixup=0.1,

            # Training settings
            resume=resume_stage1,  # Resume from checkpoint if available
            save=True,
            save_period=-1,  # Only save best and last
            project='runs/potholed',
            name='stage1_640px',
            exist_ok=True,
            verbose=True,
        )

        # Get Stage 1 best model
        best_stage1_path = Path(model.trainer.save_dir) / "weights" / "best.pt"

        print("\n" + "=" * 80)
        print("✅ Stage 1 Complete!")
        print("=" * 80)
        print(f"Best Stage 1 model: {best_stage1_path}")

        # Print Stage 1 metrics
        if hasattr(results_stage1, 'results_dict'):
            metrics = results_stage1.results_dict
            print("\nStage 1 metrics (at 640px):")
            if 'metrics/precision(B)' in metrics:
                print(f"  Precision: {metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics:
                print(f"  Recall: {metrics['metrics/recall(B)']:.4f}")
            if 'metrics/mAP50(B)' in metrics:
                print(f"  mAP@50: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"  mAP@50-95: {metrics['metrics/mAP50-95(B)']:.4f}")

    # =========================================================================
    # STAGE 2: Fine-tuning at Deployment Resolution (256px)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: Fine-tuning at 256px for deployment...")
    print("=" * 80)

    # Check if Stage 2 can be resumed or is already complete
    stage2_dir = Path('runs/potholed/stage2_256px')
    stage2_last = stage2_dir / 'weights' / 'last.pt'
    stage2_best = stage2_dir / 'weights' / 'best.pt'

    if stage2_best.exists():
        print(f"\n✅ Stage 2 already completed! Found best model: {stage2_best}")
        print("  Training is complete!")
        best_model_path = stage2_best
        skip_stage2 = True
    elif stage2_last.exists():
        print(f"\n⚠️ Found Stage 2 checkpoint: {stage2_last}")
        print("  Resuming Stage 2 training from last checkpoint...")
        model = YOLO(str(stage2_last))
        skip_stage2 = False
        resume_stage2 = True
    else:
        # Load Stage 1 best weights
        print(f"\nLoading Stage 1 best weights: {best_stage1_path}")
        model = YOLO(str(best_stage1_path))
        skip_stage2 = False
        resume_stage2 = False

    if not skip_stage2:
        print("\nStage 2 configuration:")
        print("  Model: YOLO11n (from Stage 1)")
        print("  Epochs: 100")
        print("  Image size: 256x256 (deployment resolution, matches cropped input)")
        print("  Batch size: 32")
        print("  Optimizer: SGD")
        print("  Learning rate: 0.001 (10x lower for fine-tuning)")
        print("  Momentum: 0.937")
        print("  Weight decay: 0.0005")
        print("  Data augmentation (moderate - balanced for fine-tuning):")
        print("    - Horizontal flip: 0.5")
        print("    - Vertical flip: 0.5")
        print("    - Rotation: ±5°")
        print("    - Shear: ±1.5°")
        print("    - HSV augmentation: h=0.015, s=0.6, v=0.4")
        print("    - Scale: 0.5")
        print("    - Mosaic: 0.5")
        if resume_stage2:
            print("\n⚠️ RESUMING Stage 2 training from checkpoint...\n")
        else:
            print("\nStarting Stage 2 fine-tuning...\n")

        # Train Stage 2
        results_stage2 = model.train(
            data=str(data_yaml_path),
            epochs=100,
            imgsz=256,
            batch=32,
            optimizer='SGD',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,

            # Moderate augmentation for fine-tuning
            hsv_h=0.015,
            hsv_s=0.6,
            hsv_v=0.4,
            fliplr=0.5,
            flipud=0.5,
            degrees=5.0,
            shear=1.5,
            scale=0.5,
            mosaic=0.5,

            # Training settings
            resume=resume_stage2,  # Resume from checkpoint if available
            save=True,
            save_period=-1,  # Only save best and last
            project='runs/potholed',
            name='stage2_256px',
            exist_ok=True,
            verbose=True,
        )

        # Get final best model
        best_model_path = Path(model.trainer.save_dir) / "weights" / "best.pt"

    print("\n" + "=" * 80)
    print("STEP 5: Two-Stage Training Complete!")
    print("=" * 80)

    print(f"\n✅ Stage 1 model (640px): {best_stage1_path}")
    print(f"✅ Stage 2 model (256px): {best_model_path}")
    print(f"✅ Final model for deployment: {best_model_path}")

    # Print final metrics
    if not skip_stage2 and hasattr(results_stage2, 'results_dict'):
        metrics = results_stage2.results_dict
        print("\nFinal metrics (Stage 2 at 256px):")
        if 'metrics/precision(B)' in metrics:
            print(f"  Precision: {metrics['metrics/precision(B)']:.4f}")
        if 'metrics/recall(B)' in metrics:
            print(f"  Recall: {metrics['metrics/recall(B)']:.4f}")
        if 'metrics/mAP50(B)' in metrics:
            print(f"  mAP@50: {metrics['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"  mAP@50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
    elif skip_stage2:
        print("\n(Stage 2 was already complete - metrics available in training logs)")

    print("\nTarget benchmarks from paper (YOLOv9-c at 640px):")
    print("  Precision: 0.92")
    print("  Recall: 0.90")
    print("  mAP@50: 0.95")
    print("  mAP@50-95: 0.64")
    print("  F1-score: 0.91")

    print("\nTwo-stage training benefits (with improvements):")
    print("  ✅ Rich feature learning from 640px training (200 epochs)")
    print("  ✅ Optimized for 256px deployment (matches cropped input, faster inference)")
    print("  ✅ Better small object detection")
    print("  ✅ Stronger augmentation increases generalization")
    print("  ✅ Expected improvements: +10-15% mAP vs previous training")

    print("\nNext steps:")
    print("  1. Evaluate the model on test data")
    print("  2. Export to ONNX format for TinyGrad conversion")
    print("  3. Convert to TinyGrad pkl for deployment in potholed.py")

    return best_model_path


def export_onnx(model_path: Path, output_dir: Path | None = None, imgsz: int = 256, half: bool = True) -> Path:
    """
    Export trained model to ONNX (FP16 by default) and return the path.

    Args:
        model_path: Path to the trained .pt model
        output_dir: Output directory for ONNX file (default: runs/potholed/stage3_export)
        imgsz: Image size for export (default: 320)
        half: If True, export FP16 ONNX; if False, export FP32 ONNX (default: True)
    """
    precision = "FP16" if half else "FP32"
    print("\n" + "=" * 80)
    print(f"Exporting model to {precision} ONNX")
    print("=" * 80)
    print(f"\nSource model: {model_path}")

    # Determine output directory
    if output_dir is None:
        output_dir = Path('runs/potholed/stage3_export')
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    export_result = model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=True,
        dynamic=False,
        opset=12,
        half=half,  # FP16 export
        verbose=False,
        project=str(output_dir),
        name=f'onnx_{precision.lower()}'
    )

    if isinstance(export_result, (str, Path)):
        onnx_path = Path(export_result)
    else:
        onnx_path = output_dir / f'onnx_{precision.lower()}' / 'model.onnx'

    if not onnx_path.exists():
        alt_path = output_dir / f'onnx_{precision.lower()}.onnx'
        if alt_path.exists():
            onnx_path = alt_path
        else:
            print(f"ERROR: ONNX export not found at {onnx_path}")
            sys.exit(1)

    print(f"✅ ONNX ({precision}) saved: {onnx_path}")
    return onnx_path

def main():
    """Main training pipeline"""
    print("\n" + "=" * 80)
    print("YOLO11n Pothole Detection Training")
    print("Dataset: Multi-Weather Pothole Detection (MWPD)")
    print("=" * 80)

    # Set working directory to script location
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent

    # Download dataset
    dataset_path = download_dataset()

    # Analyze dataset structure
    structure = find_dataset_structure(dataset_path)

    if structure['format'] is None:
        print("\nERROR: Could not determine dataset format")
        sys.exit(1)

    # Prepare YOLO dataset
    data_yaml_path = create_yolo_dataset(dataset_path, structure, workspace_root / "data")

    # Train model (two-stage training)
    best_model_path = train_yolo11n(data_yaml_path)

    print("\n" + "=" * 80)
    print("✅ TWO-STAGE TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest model: {best_model_path}")

    # Export Stage 2 best model to FP16 ONNX for TinyGrad pipeline (FP16 for better performance)
    print("\n" + "=" * 80)
    print("Exporting Stage 2 best model to FP16 ONNX...")
    print("=" * 80)
    onnx_fp16_path = export_onnx(best_model_path, imgsz=256, half=True)
    print(f"\nFP16 ONNX model: {onnx_fp16_path}")

    # Copy the ONNX artifact into laica/potholed/models directory (do not delete existing models)
    models_dir = script_dir.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    dest_path = models_dir / Path(onnx_fp16_path).name
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

