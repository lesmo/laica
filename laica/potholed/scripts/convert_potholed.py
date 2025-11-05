#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Allow running from anywhere in the repo
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tinygrad_repo'))

MODELS_DIR = REPO_ROOT / 'laica' / 'potholed' / 'models'


def convert_onnx_to_pkl(onnx_path: Path, output_path: Path, use_cuda: bool = False, no_validate: bool = False) -> bool:
  """Convert a single ONNX file to PKL format."""
  try:
    # Configure tinygrad backend
    import os
    import pickle

    # Auto-detect TICI (Comma 3X) for QCOM/OpenCL
    # Try multiple methods for robust detection
    is_tici = False
    try:
      from openpilot.system.hardware import TICI
      is_tici = TICI
    except (ImportError, AttributeError):
      pass

    # Fallback: check for /TICI file directly (more reliable)
    if not is_tici:
      is_tici = os.path.isfile('/TICI')

    # Also check environment variable as override
    if 'DEV' in os.environ and os.environ['DEV'] == 'QCOM':
      is_tici = True  # If DEV=QCOM is set, assume TICI

    if 'DEV' not in os.environ:
      if use_cuda:
        os.environ['DEV'] = 'CUDA'
      elif is_tici:
        os.environ['DEV'] = 'QCOM'  # OpenCL for C3X
        print(f"✓ Detected TICI (C3X), using QCOM/OpenCL backend for {onnx_path.name}")
      else:
        os.environ['DEV'] = 'LLVM'  # CPU fallback
    if 'FLOAT16' not in os.environ:
      os.environ['FLOAT16'] = '1'
    if 'IMAGE' not in os.environ:
      os.environ['IMAGE'] = '0'
    if 'NOLOCALS' not in os.environ:
      os.environ['NOLOCALS'] = '1'
    if 'JIT_BATCH_SIZE' not in os.environ:
      os.environ['JIT_BATCH_SIZE'] = '0'

    from tinygrad import Tensor, TinyJit, Context, GlobalCounters, Device
    from tinygrad.helpers import DEBUG
    from tinygrad.engine.realize import CompiledRunner
    from tinygrad.frontend.onnx import OnnxRunner

    # Load ONNX
    run_onnx = OnnxRunner(str(onnx_path))

    # Derive inputs - preserve original ONNX input types for better FP16 optimization
    # With FLOAT16=1, TinyGrad can handle float16 inputs natively, reducing memory bandwidth
    input_shapes = {name: spec.shape for name, spec in run_onnx.graph_inputs.items()}
    input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}
    # Note: Removed float16->float32 cast to enable full FP16 optimization
    # If ONNX has float16 inputs, they will be preserved

    Tensor.manual_seed(100)
    # Create inputs on the target device (Device.DEFAULT) so JIT captures the correct device
    target_device = Device.DEFAULT
    print(f"✓ Compiling {onnx_path.name} for device: {target_device}")
    new_inputs = {k: Tensor.randn(*shp, dtype=input_types[k], device=target_device).mul(8).realize() for k, shp in sorted(input_shapes.items())}
    new_inputs_numpy = {k: v.numpy() for k, v in new_inputs.items()}

    # Create JIT - inputs should already be on Device.DEFAULT, but ensure they are
    run_onnx_jit = TinyJit(
      lambda **kwargs: next(iter(run_onnx({k: v.to(Device.DEFAULT) for k, v in kwargs.items()}).values())).cast('float32'),
      prune=True,
    )

    # Optional validation
    if not no_validate:
      for i in range(3):
        GlobalCounters.reset()
        inputs = {
          **{k: v.clone() for k, v in new_inputs.items() if 'img' in k},
          **{k: Tensor(v, device=target_device).realize() for k, v in new_inputs_numpy.items() if 'img' not in k},
        }
        with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
          _ = run_onnx_jit(**inputs).numpy()

      # Kernel checks (non-fatal, but useful)
      kernel_count = 0
      read_image_count = 0
      gated_read_image_count = 0
      for ei in run_onnx_jit.captured.jit_cache:
        if isinstance(ei.prg, CompiledRunner):
          kernel_count += 1
          ps = ei.prg.p.src
          read_image_count += ps.count('read_image')
          gated_read_image_count += ps.count('?read_image')

    # Write PKL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
      pickle.dump(run_onnx_jit, f)

    print(f"✓ Success: wrote {output_path}")
    return True
  except Exception as e:
    print(f"✗ Conversion failed for {onnx_path.name}: {e}")
    import traceback
    traceback.print_exc()
    return False


def main() -> int:
  parser = argparse.ArgumentParser(description='Convert all pothole ONNX models to Tinygrad PKL')
  parser.add_argument('--cuda', action='store_true', help='Use CUDA converter')
  parser.add_argument('--no-validate', action='store_true', help='Skip validation step')

  args = parser.parse_args()

  models_dir = MODELS_DIR

  if not models_dir.exists():
    print(f"Error: Models directory not found: {models_dir}")
    return 1

  # Find all ONNX files
  onnx_files = sorted(models_dir.glob('*.onnx'))

  if not onnx_files:
    print(f"No ONNX files found in {models_dir}")
    return 1

  print(f"Found {len(onnx_files)} ONNX file(s) to convert:")
  for onnx_file in onnx_files:
    print(f"  - {onnx_file.name}")

  print("\n" + "=" * 80)

  success_count = 0
  for onnx_path in onnx_files:
    # Output PKL to same directory with .pkl extension
    output_path = onnx_path.with_suffix('.pkl')

    print(f"\nConverting: {onnx_path.name} -> {output_path.name}")
    print("-" * 80)

    if convert_onnx_to_pkl(onnx_path, output_path, args.cuda, args.no_validate):
      success_count += 1

  print("\n" + "=" * 80)
  print(f"Conversion complete: {success_count}/{len(onnx_files)} successful")
  print("=" * 80)

  return 0 if success_count == len(onnx_files) else 1


if __name__ == '__main__':
  raise SystemExit(main())


