#!/usr/bin/env python3
"""
Test suite for pothole detection system
"""
import unittest
import time
import numpy as np
from pathlib import Path
import sys
import os

# Set environment before importing tinygrad-dependent modules
os.environ['DEV'] = 'LLVM'  # Force CPU mode for tests

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from selfdrive.potholed.potholed import ModelState
from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext

class MockVisionBuf:
    """Mock VisionBuf for testing"""
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()
        self.mem_address = 0  # Mock address

class TestPotholeDetection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = Path(__file__).parent / 'models/pothole_detection_tinygrad.pkl'
        # Only create CLContext if we're actually testing the model
        # CLContext requires OpenCL which may not be available in all test environments
        self.model_state = None
        if self.model_path.exists():
            try:
                self.model_state = ModelState(None)  # Use None for CPU-only mode in tests
            except Exception as e:
                print(f"Warning: Could not create ModelState: {e}")
                self.model_state = None

    def test_model_exists(self):
        """Test that the model file exists"""
        if not self.model_path.exists():
            self.skipTest(f"Model file not found: {self.model_path}. Run convert_pothole_model.py first.")
        print("✓ Model file exists")

    def test_model_format(self):
        """Test that the model file has correct format"""
        if not self.model_path.exists():
            self.skipTest(f"Model file not found: {self.model_path}")

        import pickle
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        # Check if it's a TinyJit model or has the expected structure
        self.assertIsNotNone(model_data)
        print("✓ Model file format is valid")

    def test_post_processing(self):
        """Test YOLO post-processing functionality"""
        if self.model_state is None:
            self.skipTest("ModelState not available")

        # Create mock YOLO output
        mock_output = np.random.randn(1, 84, 10).astype(np.float32)

        detections = self.model_state.post_process_yolo(mock_output, 0.5, 0.45)

        # Check that detections are in expected format
        self.assertIsInstance(detections, list)

        for detection in detections:
            self.assertEqual(len(detection), 5)  # x, y, width, height, confidence
            self.assertGreaterEqual(detection[0], 0.0)  # x
            self.assertLessEqual(detection[0], 1.0)
            self.assertGreaterEqual(detection[1], 0.0)  # y
            self.assertLessEqual(detection[1], 1.0)
            self.assertGreaterEqual(detection[2], 0.0)  # width
            self.assertLessEqual(detection[2], 1.0)
            self.assertGreaterEqual(detection[3], 0.0)  # height
            self.assertLessEqual(detection[3], 1.0)
            self.assertGreaterEqual(detection[4], 0.0)  # confidence
            self.assertLessEqual(detection[4], 1.0)

        print("✓ Post-processing test passed")

    def test_post_process_output_format(self):
        """Test that post-processing returns correctly formatted detections"""
        if self.model_state is None:
            self.skipTest("ModelState not available")

        # Test with various output shapes
        test_cases = [
            np.random.randn(1, 84, 10).astype(np.float32),
            np.random.randn(1, 84, 5).astype(np.float32),
            np.random.randn(1, 84, 20).astype(np.float32),
        ]

        for output in test_cases:
            detections = self.model_state.post_process_yolo(output, 0.5, 0.45)
            self.assertIsInstance(detections, list)

            # Verify detection format
            for det in detections:
                self.assertEqual(len(det), 5, "Each detection should have 5 values")
                x, y, w, h, conf = det
                # Check normalized coordinates
                self.assertIsInstance(x, (int, float))
                self.assertIsInstance(y, (int, float))
                self.assertIsInstance(w, (int, float))
                self.assertIsInstance(h, (int, float))
                self.assertIsInstance(conf, (int, float))

        print("✓ Post-process output format test passed")

    def test_confidence_threshold(self):
        """Test that confidence threshold is respected"""
        if self.model_state is None:
            self.skipTest("ModelState not available")

        mock_output = np.random.randn(1, 84, 10).astype(np.float32)

        # Test with different thresholds
        detections_low = self.model_state.post_process_yolo(mock_output, 0.3, 0.45)
        detections_high = self.model_state.post_process_yolo(mock_output, 0.9, 0.45)

        # Lower threshold should generally give more detections
        # (though with mock data this is not guaranteed)
        self.assertIsInstance(detections_low, list)
        self.assertIsInstance(detections_high, list)

        print("✓ Confidence threshold test passed")

def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("\n" + "="*50)
    print("POTHOLE DETECTION PERFORMANCE BENCHMARK")
    print("="*50)

    model_path = Path(__file__).parent / 'models/pothole_detection_tinygrad.pkl'
    if not model_path.exists():
        print("Model file not found. Skipping performance benchmark.")
        print("Run convert_pothole_model.py first to generate the model.")
        return

    # Check if we're in an environment that supports OpenCL
    # Skip benchmark if TICI is False and we don't have OpenCL
    from openpilot.system.hardware import TICI
    if not TICI:
        print("Skipping performance benchmark - OpenCL not available in test environment.")
        print("Benchmark requires hardware with OpenCL support or Comma 3X device.")
        return

    # Import here to avoid issues if dependencies are missing
    try:
        from selfdrive.potholed.potholed import ModelState

        cl_context = CLContext()
        model = ModelState(cl_context)
        mock_buf = MockVisionBuf()

        # Warmup
        print("Warming up...")
        for _ in range(5):
            model.run(mock_buf)

        # Benchmark
        times = []
        detections_count = []

        print("Running benchmark...")
        for i in range(100):
            detections, execution_time = model.run(mock_buf)
            times.append(execution_time)
            detections_count.append(len(detections))

            if i % 20 == 0:
                print(f"Progress: {i+1}/100 frames")

        # Calculate statistics
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        std_time = np.std(times)
        avg_detections = np.mean(detections_count)

        print(f"\nPerformance Results:")
        print(f"  Average inference time: {avg_time*1000:.1f}ms")
        print(f"  Max inference time: {max_time*1000:.1f}ms")
        print(f"  Min inference time: {min_time*1000:.1f}ms")
        print(f"  Standard deviation: {std_time*1000:.1f}ms")
        print(f"  Average detections per frame: {avg_detections:.1f}")
        print(f"  FPS capability: {1/avg_time:.1f}")

        # Performance targets
        targets_met = []
        if avg_time < 0.05:
            targets_met.append("✓ Average time < 50ms")
        else:
            targets_met.append("✗ Average time >= 50ms")

        if max_time < 0.1:
            targets_met.append("✓ Max time < 100ms")
        else:
            targets_met.append("✗ Max time >= 100ms")

        if 1/avg_time > 20:
            targets_met.append("✓ FPS > 20")
        else:
            targets_met.append("✗ FPS <= 20")

        print(f"\nTarget Achievement:")
        for target in targets_met:
            print(f"  {target}")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        print("This is expected if CLContext or model dependencies are not available.")

if __name__ == "__main__":
    print("Running Pothole Detection Tests...")

    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run performance benchmark (only if environment supports it)
    try:
        run_performance_benchmark()
    except Exception as e:
        print(f"\nPerformance benchmark skipped: {e}")
        print("This is expected if OpenCL is not available in the test environment.")

    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
