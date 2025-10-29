#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Allow running from anywhere in the repo
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tinygrad_repo'))

DEFAULT_MODELS_DIR = REPO_ROOT / 'laica' / 'potholed' / 'models'
DEFAULT_ONNX = DEFAULT_MODELS_DIR / 'best.onnx'
DEFAULT_PKL = DEFAULT_MODELS_DIR / 'best.pkl'


def main() -> int:
  parser = argparse.ArgumentParser(description='Convert pothole ONNX model to Tinygrad PKL')
  parser.add_argument('--onnx', type=str, default=str(DEFAULT_ONNX), help='Path to ONNX model')
  parser.add_argument('--output', type=str, help='Path to output PKL (auto by backend if omitted)')
  parser.add_argument('--cuda', action='store_true', help='Use CUDA converter')
  parser.add_argument('--no-validate', action='store_true', help='Skip validation step')

  args = parser.parse_args()

  onnx_path = Path(args.onnx)

  if not onnx_path.exists():
    print(f"Error: ONNX file not found: {onnx_path}")
    if DEFAULT_MODELS_DIR.exists():
      print("Available ONNX files:")
      for p in sorted(DEFAULT_MODELS_DIR.glob('*.onnx')):
        print(f"  {p}")
    return 1

  # Select default output if not provided
  output_path = Path(args.output) if args.output else DEFAULT_PKL
  output_path.parent.mkdir(parents=True, exist_ok=True)

  try:
    # Configure tinygrad backend
    import os
    import pickle
    if 'DEV' not in os.environ:
      os.environ['DEV'] = 'CUDA' if args.cuda else 'LLVM'
    if 'FLOAT16' not in os.environ:
      os.environ['FLOAT16'] = '1'
    if 'IMAGE' not in os.environ:
      os.environ['IMAGE'] = '0'
    if 'NOLOCALS' not in os.environ:
      os.environ['NOLOCALS'] = '1'
    if 'JIT_BATCH_SIZE' not in os.environ:
      os.environ['JIT_BATCH_SIZE'] = '0'

    from tinygrad import Tensor, TinyJit, Context, GlobalCounters, Device, dtypes
    from tinygrad.helpers import DEBUG
    from tinygrad.engine.realize import CompiledRunner
    from tinygrad.frontend.onnx import OnnxRunner

    # Load ONNX
    run_onnx = OnnxRunner(str(onnx_path))

    # Derive inputs
    input_shapes = {name: spec.shape for name, spec in run_onnx.graph_inputs.items()}
    input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}
    input_types = {k: (dtypes.float32 if v is dtypes.float16 else v) for k, v in input_types.items()}

    Tensor.manual_seed(100)
    new_inputs = {k: Tensor.randn(*shp, dtype=input_types[k]).mul(8).realize() for k, shp in sorted(input_shapes.items())}
    new_inputs_numpy = {k: v.numpy() for k, v in new_inputs.items()}

    # Create JIT
    run_onnx_jit = TinyJit(
      lambda **kwargs: next(iter(run_onnx({k: v.to(Device.DEFAULT) for k, v in kwargs.items()}).values())).cast('float32'),
      prune=True,
    )

    # Optional validation
    if not args.no_validate:
      for i in range(3):
        GlobalCounters.reset()
        inputs = {
          **{k: v.clone() for k, v in new_inputs.items() if 'img' in k},
          **{k: Tensor(v, device='NPY').realize() for k, v in new_inputs_numpy.items() if 'img' not in k},
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
    with open(output_path, 'wb') as f:
      pickle.dump(run_onnx_jit, f)
  except Exception as e:
    print(f"Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    return 1

  print(f"\nSuccess: wrote {output_path}")
  return 0


if __name__ == '__main__':
  raise SystemExit(main())


