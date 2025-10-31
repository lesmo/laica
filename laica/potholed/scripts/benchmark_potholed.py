#!/usr/bin/env python3
"""
TinyGrad .pkl Pothole Detection Benchmark Script
Tests inference speed under Comma 3X resource constraints

Simulates:
- CPU-only inference (Comma 3X uses Snapdragon 845, 8 cores)
- Limited memory (6GB on Comma 3X)
- Real-world camera resolution (1928x1208 YUV420)
- Concurrent process load (other openpilot processes)
"""
import os
import sys
import time
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Ensure repo paths are on sys.path (for tinygrad_repo and local modules)
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TG_PATH = REPO_ROOT / 'tinygrad_repo'
if str(TG_PATH) not in sys.path:
    sys.path.insert(0, str(TG_PATH))

# Set CPU-only for realistic Comma 3X simulation
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Auto-detect TICI (Comma 3X) for device selection
try:
    from openpilot.system.hardware import TICI
    IS_TICI = TICI
except (ImportError, AttributeError):
    IS_TICI = False

# Set TinyGrad device based on platform
if IS_TICI:
    os.environ['DEV'] = 'QCOM'  # OpenCL for C3X
    DEFAULT_DEVICE = 'GPU'  # TinyGrad will use OpenCL/GPU
    print("✓ Detected TICI (C3X), using QCOM/OpenCL backend")
else:
    os.environ['DEV'] = 'LLVM'
    DEFAULT_DEVICE = 'NPY'  # CPU fallback
    print("✓ Using CPU (NPY) backend")

# Enable FP16 optimization for better performance on GPU
if 'FLOAT16' not in os.environ:
    os.environ['FLOAT16'] = '1'
    print("✓ Enabled FP16 optimization")


def set_cpu_affinity(num_cores: int = 4):
    """Limit to specific CPU cores to simulate Comma 3X constraints"""
    try:
        import psutil
        p = psutil.Process()
        # Use only first N cores (Comma 3X has 8 cores, but reserve some for other processes)
        available_cores = list(range(min(num_cores, psutil.cpu_count())))
        p.cpu_affinity(available_cores)
        print(f"✓ CPU affinity set to {len(available_cores)} cores: {available_cores}")
    except Exception as e:
        print(f"⚠ Could not set CPU affinity: {e}")


def simulate_background_load(duration: float, target_cpu_percent: float = 30):
    """Simulate background CPU load from other openpilot processes"""
    import multiprocessing

    def cpu_load():
        end_time = time.time() + duration
        while time.time() < end_time:
            # Busy loop to consume CPU
            _ = sum(i*i for i in range(10000))

    num_workers = max(1, int(multiprocessing.cpu_count() * target_cpu_percent / 100))
    workers = []

    for _ in range(num_workers):
        p = multiprocessing.Process(target=cpu_load)
        p.start()
        workers.append(p)

    return workers


def generate_test_images(count: int = 100, size: Tuple[int, int] = (1928, 1208)) -> List[np.ndarray]:
    """Generate synthetic test images matching Comma 3X camera resolution"""
    print(f"\nGenerating {count} test images ({size[0]}x{size[1]})...")
    images = []

    for i in range(count):
        # Generate random RGB image
        img = np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)
        images.append(img)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{count} images...")

    print(f"✓ Generated {count} test images")
    return images


def benchmark_model(model_path: Path, images: List[np.ndarray],
                   target_size: int = 320, warmup_runs: int = 10, device: str = None) -> Dict:
    """Benchmark TinyGrad .pkl model inference speed"""

    if device is None:
        device = DEFAULT_DEVICE

    import pickle
    from tinygrad.tensor import Tensor
    from tinygrad.dtype import dtypes

    print(f"\n{'='*80}")
    print(f"Loading model: {model_path}")
    print(f"{'='*80}")

    # Load model_run from pickle (supports dict format with metadata or direct TinyGrad function)
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    if isinstance(model_data, dict) and 'model_run' in model_data:
        model_run = model_data['model_run']
        metadata = model_data.get('metadata', {})
    else:
        model_run = model_data
        metadata = {'model_type': 'tinygrad_jit'}

    # Detect input dtype from compiled model (supports FP16 from ONNX export)
    input_dtype = dtypes.float32  # Default fallback
    try:
        # Try to get input dtype from model's captured function
        if hasattr(model_run, 'captured') and hasattr(model_run.captured, 'expected_st_vars_dtype_device'):
            # Find 'images' input in the expected inputs
            if hasattr(model_run.captured, 'expected_names'):
                for idx, name in enumerate(model_run.captured.expected_names):
                    if 'images' in name or 'img' in name or name == 'images':
                        if idx < len(model_run.captured.expected_st_vars_dtype_device):
                            input_dtype = model_run.captured.expected_st_vars_dtype_device[idx][2]  # dtype
                            break
    except (AttributeError, IndexError):
        pass  # Fall back to float32

    print(f"Model type: {metadata.get('model_type', 'unknown')}")
    print(f"Device: {device}")
    print(f"Input dtype: {input_dtype}")
    print(f"Target inference size: {target_size}x{target_size}")
    print(f"Test images: {len(images)}")
    print(f"Warmup runs: {warmup_runs}")

    # Helper: preprocess np.ndarray RGB image -> CHW batch of size 1
    # Use numpy dtype that matches the model's expected input dtype
    numpy_dtype = np.float32 if input_dtype == dtypes.float32 else np.float16

    def preprocess(img: np.ndarray) -> np.ndarray:
        import cv2
        img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img_normalized = img_resized.astype(numpy_dtype) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0)

    # Check if FP16 is enabled for internal operations
    use_fp16_internals = os.environ.get('FLOAT16') == '1'
    dtype_str = f"{input_dtype} inputs"
    if use_fp16_internals and input_dtype == dtypes.float32:
        dtype_str += " + FP16 internals"
    elif input_dtype == dtypes.float16:
        dtype_str += " (full FP16)"

    # Warmup
    print(f"\nWarming up model ({warmup_runs} runs) on {device} ({dtype_str})...")
    for i in range(warmup_runs):
        np_in = preprocess(images[i % len(images)])
        inp = Tensor(np_in, dtype=input_dtype, device=device)
        _ = model_run(images=inp)
    print("✓ Warmup complete")

    # Benchmark
    print(f"\nRunning benchmark on {len(images)} images...")
    inference_times = []
    total_times = []

    for i, img in enumerate(images):
        t_start = time.perf_counter()

        # Inference
        t_inference_start = time.perf_counter()
        np_in = preprocess(img)
        inp = Tensor(np_in, dtype=input_dtype, device=device)
        _ = model_run(images=inp)
        t_inference_end = time.perf_counter()

        t_end = time.perf_counter()

        inference_time = t_inference_end - t_inference_start
        total_time = t_end - t_start

        inference_times.append(inference_time)
        total_times.append(total_time)

        if (i + 1) % 20 == 0:
            avg_inference = np.mean(inference_times[-20:]) * 1000
            avg_total = np.mean(total_times[-20:]) * 1000
            print(f"  {i+1}/{len(images)}: inference={avg_inference:.1f}ms, total={avg_total:.1f}ms")

    # Calculate statistics
    inference_times_ms = np.array(inference_times) * 1000
    total_times_ms = np.array(total_times) * 1000

    stats = {
        'model_path': str(model_path),
        'num_images': len(images),
        'target_size': target_size,

        # Inference timing
        'inference_mean_ms': float(np.mean(inference_times_ms)),
        'inference_std_ms': float(np.std(inference_times_ms)),
        'inference_min_ms': float(np.min(inference_times_ms)),
        'inference_max_ms': float(np.max(inference_times_ms)),
        'inference_p50_ms': float(np.percentile(inference_times_ms, 50)),
        'inference_p95_ms': float(np.percentile(inference_times_ms, 95)),
        'inference_p99_ms': float(np.percentile(inference_times_ms, 99)),

        # Total timing (includes preprocessing)
        'total_mean_ms': float(np.mean(total_times_ms)),
        'total_std_ms': float(np.std(total_times_ms)),
        'total_p95_ms': float(np.percentile(total_times_ms, 95)),

        # FPS calculations
        'fps_mean': float(1000 / np.mean(total_times_ms)),
        'fps_p95': float(1000 / np.percentile(total_times_ms, 95)),
    }

    return stats


def print_benchmark_results(stats: Dict, target_fps: int = 20):
    """Print formatted benchmark results"""
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")

    print(f"\nModel: {Path(stats['model_path']).name}")
    print(f"Input size: {stats['target_size']}x{stats['target_size']}")
    print(f"Test images: {stats['num_images']}")

    print(f"\n{'─'*80}")
    print("INFERENCE TIME (pure model execution)")
    print(f"{'─'*80}")
    print(f"  Mean:       {stats['inference_mean_ms']:7.2f} ms  (±{stats['inference_std_ms']:.2f})")
    print(f"  Min:        {stats['inference_min_ms']:7.2f} ms")
    print(f"  Max:        {stats['inference_max_ms']:7.2f} ms")
    print(f"  Median:     {stats['inference_p50_ms']:7.2f} ms")
    print(f"  95th %ile:  {stats['inference_p95_ms']:7.2f} ms")
    print(f"  99th %ile:  {stats['inference_p99_ms']:7.2f} ms")

    print(f"\n{'─'*80}")
    print("TOTAL TIME (inference + preprocessing)")
    print(f"{'─'*80}")
    print(f"  Mean:       {stats['total_mean_ms']:7.2f} ms  (±{stats['total_std_ms']:.2f})")
    print(f"  95th %ile:  {stats['total_p95_ms']:7.2f} ms")

    print(f"\n{'─'*80}")
    print("FRAMES PER SECOND (FPS)")
    print(f"{'─'*80}")
    print(f"  Mean FPS:   {stats['fps_mean']:7.2f} fps")
    print(f"  95th %ile:  {stats['fps_p95']:7.2f} fps")

    # Target assessment
    target_time_ms = 1000 / target_fps
    print(f"\n{'─'*80}")
    print(f"TARGET: {target_fps} FPS (≤{target_time_ms:.1f}ms per frame)")
    print(f"{'─'*80}")

    if stats['total_p95_ms'] <= target_time_ms:
        print(f"  ✅ PASSED: 95th percentile ({stats['total_p95_ms']:.1f}ms) meets target")
    else:
        overhead = stats['total_p95_ms'] - target_time_ms
        print(f"  ❌ FAILED: 95th percentile ({stats['total_p95_ms']:.1f}ms) exceeds target by {overhead:.1f}ms")

    if stats['total_mean_ms'] <= target_time_ms:
        print(f"  ✅ PASSED: Mean ({stats['total_mean_ms']:.1f}ms) meets target")
    else:
        overhead = stats['total_mean_ms'] - target_time_ms
        print(f"  ❌ FAILED: Mean ({stats['total_mean_ms']:.1f}ms) exceeds target by {overhead:.1f}ms")

    # Real-time capability assessment
    print(f"\n{'─'*80}")
    print("REAL-TIME CAPABILITY")
    print(f"{'─'*80}")

    if stats['fps_p95'] >= target_fps:
        print(f"  ✅ Model can sustain {target_fps} FPS (95th %ile: {stats['fps_p95']:.1f} fps)")
    else:
        deficit = target_fps - stats['fps_p95']
        print(f"  ⚠️  Model may struggle at {target_fps} FPS (95th %ile: {stats['fps_p95']:.1f} fps, short by {deficit:.1f} fps)")

    print(f"\n{'='*80}\n")


def compare_models(model_paths: List[Path], images: List[np.ndarray], target_size: int = 320, device: str = None):
    """Compare multiple models"""
    results = []

    for model_path in model_paths:
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            continue

        stats = benchmark_model(model_path, images, target_size, device=device)
        results.append(stats)

    # Print comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}\n")

        print(f"{'Model':<40} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Mean FPS':<12}")
        print(f"{'-'*40} {'-'*12} {'-'*12} {'-'*12}")

        for stats in results:
            model_name = Path(stats['model_path']).stem
            print(f"{model_name:<40} {stats['total_mean_ms']:>10.2f}  {stats['total_p95_ms']:>10.2f}  {stats['fps_mean']:>10.2f}")

        print()

    return results


def main():
    """Main benchmark routine"""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark TinyGrad .pkl pothole detection for Comma 3X')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to compiled TinyGrad model (.pkl file)')
    parser.add_argument('--size', type=int, default=320,
                       help='Inference size (default: 320)')
    parser.add_argument('--images', type=int, default=100,
                       help='Number of test images (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Warmup runs (default: 10)')
    parser.add_argument('--cores', type=int, default=4,
                       help='CPU cores to use (default: 4, simulating Comma 3X constraints)')
    parser.add_argument('--target-fps', type=int, default=20,
                       help='Target FPS (default: 20)')
    parser.add_argument('--background-load', type=float, default=0,
                       help='Simulate background CPU load (0-100%, default: 0)')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple models')
    parser.add_argument('--device', type=str, default=None,
                       help=f'Override device (default: auto-detect, current: {DEFAULT_DEVICE})')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("YOLO11n Pothole Detection Benchmark")
    print("Comma 3X Resource Constraints Simulation")
    print(f"{'='*80}\n")

    # System info
    print("System Information:")
    print(f"  CPU cores available: {psutil.cpu_count()}")
    print(f"  CPU cores to use: {args.cores}")
    print(f"  RAM available: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  Target FPS: {args.target_fps}")
    print(f"  Background load: {args.background_load}%")

    # Set CPU constraints
    set_cpu_affinity(args.cores)

    # Start background load if requested
    background_workers = []
    if args.background_load > 0:
        print(f"\n⚠️  Starting background load simulation ({args.background_load}% CPU)...")
        background_workers = simulate_background_load(duration=999999, target_cpu_percent=args.background_load)

    try:
        # Generate test images
        images = generate_test_images(args.images, size=(1928, 1208))

        # Override device if specified
        device = args.device if args.device else DEFAULT_DEVICE
        if args.device:
            print(f"\n⚠️  Overriding device to: {device}")

        # Benchmark
        if args.compare:
            # Compare multiple models
            model_paths = [Path(p) for p in args.compare]
            results = compare_models(model_paths, images, args.size, device=device)

            # Print individual results
            for stats in results:
                print_benchmark_results(stats, args.target_fps)
        else:
            # Single model
            model_path = Path(args.model)
            if not model_path.exists():
                print(f"ERROR: Model not found: {model_path}")
                sys.exit(1)

            stats = benchmark_model(model_path, images, args.size, args.warmup, device=device)
            print_benchmark_results(stats, args.target_fps)

    finally:
        # Clean up background workers
        if background_workers:
            print("\nCleaning up background load simulation...")
            for p in background_workers:
                p.terminate()
                p.join()


if __name__ == "__main__":
    main()

