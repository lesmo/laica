#!/usr/bin/env python3
"""
Pothole Detection Daemon
Runs YOLO-based pothole detection on road camera feed
"""
import os
from openpilot.system.hardware import TICI
os.environ['DEV'] = 'QCOM' if TICI else 'LLVM'

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import time
import pickle
import numpy as np
from pathlib import Path
from collections import deque
import cv2

from cereal import messaging
from cereal.messaging import PubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import config_realtime_process
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext, DrivingModelFrame
from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address

# Model configuration - matches compiled model
MODEL_WIDTH, MODEL_HEIGHT = 320, 320  # Matches the compiled model input size
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 10
TOP_HALF_FILTER_ENABLED = True  # Crop input image to center region (30%-80% height) with square aspect ratio and filter large detections
CROP_START_PERCENT = 0.2  # Start cropping at 30% from top
CROP_END_PERCENT = 0.9    # End cropping at 80% from top (leaving 20% at bottom)
MAX_DETECTION_AREA = 0.1  # Maximum area a detection can cover
DEBUG_ENABLED = os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')

PROCESS_NAME = "selfdrive.potholed.potholed"
MODEL_PKL_PATH = Path(__file__).parent / 'models/pothole_detection_tinygrad.pkl'
MODEL_PKL_PATH_CUDA = Path(__file__).parent / 'models/pothole_detection_tinygrad_cuda.pkl'

class ModelState:
    def __init__(self, cl_ctx):
        self.cl_ctx = cl_ctx
        # Only create DrivingModelFrame if we have a valid CL context
        if cl_ctx is not None:
            self.frame = DrivingModelFrame(cl_ctx, 1)
        else:
            self.frame = None

        # Load model - try CUDA version first if USBGPU is set
        model_path = MODEL_PKL_PATH_CUDA if os.environ.get('USBGPU') and MODEL_PKL_PATH_CUDA.exists() else MODEL_PKL_PATH
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Handle different pickle formats
        if isinstance(model_data, dict) and 'model_run' in model_data:
            # New format with metadata
            self.model_run = model_data['model_run']
            self.metadata = model_data.get('metadata', {'model_type': 'unknown'})
        else:
            # Direct TinyJit model format
            self.model_run = model_data
            self.metadata = {'model_type': 'tinygrad_jit'}

        cloudlog.info(f"Pothole detection model loaded: {self.metadata['model_type']}")
        max_area_pct = f"{MAX_DETECTION_AREA*100:.0f}%"
        cloudlog.info(f"Input cropping enabled: {TOP_HALF_FILTER_ENABLED} (center region: 30%-80% height, square aspect ratio, max area: {max_area_pct})")

        # Detection buffer for temporal smoothing
        self.detection_buffer = deque(maxlen=5)  # 0.25s at 20Hz

        # Pre-allocate arrays for better performance
        self._debug_img_resized = None

        # Store crop parameters for coordinate mapping
        self._crop_params = None

    def preprocess_image(self, buf: VisionBuf) -> np.ndarray:
        """Preprocess camera image for YOLO input - optimized for YUV420 (Comma 3X)"""
        # Convert VisionBuf to numpy array
        img_data = np.frombuffer(buf.data, dtype=np.uint8)

        # Extract Y, U, V planes using the correct method from openpilot
        y = np.array(img_data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
        u = np.array(img_data[buf.uv_offset::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]
        v = np.array(img_data[buf.uv_offset+1::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]

        # Use the same YUV to RGB conversion as openpilot
        ul = np.repeat(np.repeat(u, 2).reshape(u.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
        vl = np.repeat(np.repeat(v, 2).reshape(v.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)

        yuv = np.dstack((y, ul, vl)).astype(np.int16)
        yuv[:, :, 1:] -= 128

        m = np.array([
            [1.00000,  1.00000, 1.00000],
            [0.00000, -0.39465, 2.03211],
            [1.13983, -0.58060, 0.00000],
        ])
        rgb_img = np.dot(yuv, m).clip(0, 255).astype(np.uint8)

        # Crop to middle portion of image to focus on road area
        if TOP_HALF_FILTER_ENABLED:
            height, width = rgb_img.shape[:2]

            # Crop vertically: from CROP_START_PERCENT to CROP_END_PERCENT from top
            start_y = int(height * CROP_START_PERCENT)
            end_y = int(height * CROP_END_PERCENT)
            cropped_height = end_y - start_y

            # Calculate target width for square aspect ratio (1:1)
            target_width = cropped_height  # Square aspect ratio

            # Center the horizontal crop
            if target_width <= width:
                # We can fit the full target width, center it
                start_x = (width - target_width) // 2
                end_x = start_x + target_width
            else:
                # Target width is larger than original, use full width
                start_x = 0
                end_x = width
                # Recalculate height to maintain square aspect ratio
                target_height = width  # Square aspect ratio
                # Adjust vertical crop to maintain square aspect ratio
                center_y = (start_y + end_y) // 2
                start_y = center_y - target_height // 2
                end_y = start_y + target_height
                cropped_height = target_height

            img_cropped = rgb_img[start_y:end_y, start_x:end_x]

            # Store crop parameters for coordinate mapping
            self._crop_params = {
                'original_height': height,
                'original_width': width,
                'start_y': start_y,
                'end_y': end_y,
                'start_x': start_x,
                'end_x': end_x,
                'cropped_height': cropped_height,
                'cropped_width': end_x - start_x
            }

            # Calculate actual crop percentages for logging
            actual_left_pct = start_x / width
            actual_right_pct = (width - end_x) / width
            actual_top_pct = start_y / height
            actual_bottom_pct = (height - end_y) / height

            crop_range_v = f"{actual_top_pct*100:.0f}%-{(1-actual_bottom_pct)*100:.0f}%"
            crop_range_h = f"{actual_left_pct*100:.0f}%-{(1-actual_right_pct)*100:.0f}%"
            debug_msg = f"Cropped image from {height}x{width} to {img_cropped.shape[0]}x{img_cropped.shape[1]} " + \
                        f"(v:{crop_range_v}, h:{crop_range_h}, square_aspect_ratio)"
            cloudlog.debug(debug_msg)
        else:
            img_cropped = rgb_img

        # Resize to model input size using OpenCV for better performance
        img_resized = cv2.resize(img_cropped, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Store the resized image for debugging detections (only if DEBUG is enabled)
        if DEBUG_ENABLED:
            # Debug: Save a sample image to see what the model is seeing
            if not hasattr(self, '_debug_image_saved'):
                cv2.imwrite('/tmp/pothole_debug_input.jpg', img_resized)
                cloudlog.info("Saved debug image to /tmp/pothole_debug_input.jpg")
                self._debug_image_saved = True

            # Store the resized image for debugging detections
            self._debug_img_resized = img_resized.copy()

        # Normalize to [0, 1] and convert to CHW format
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        img_batch = np.expand_dims(img_chw, axis=0)  # Add batch dimension

        return img_batch

    def post_process_yolo(self, output: np.ndarray, conf_threshold: float, nms_threshold: float) -> list:
        """Post-process YOLO output: apply confidence filtering and NMS"""
        detections = []

        cloudlog.debug(f"Raw model output shape: {output.shape}, dtype: {output.dtype}")
        cloudlog.debug(f"Output stats - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")

        # Handle different YOLO output formats
        if len(output.shape) == 4:  # (1, 1, 84, 8400) format
            output = output.squeeze(1)  # Remove extra dimension

        if len(output.shape) == 3:
            if output.shape[1] == 5:  # (1, 5, 8400) - single class
                predictions = output[0]  # Shape: (5, 8400)
                x_center = predictions[0]
                y_center = predictions[1]
                width = predictions[2]
                height = predictions[3]
                confidence = predictions[4]
            elif output.shape[1] == 85:  # (1, 85, 8400) - COCO format
                predictions = output[0]  # Shape: (85, 8400)
                # For single class detection, we only care about class 0 (pothole)
                x_center = predictions[0]
                y_center = predictions[1]
                width = predictions[2]
                height = predictions[3]
                confidence = predictions[4]  # Objectness score
            else:
                cloudlog.warning(f"Unexpected YOLO output shape: {output.shape}")
                return detections
        elif len(output.shape) == 2:  # (8400, 5) format
            predictions = output.T  # Transpose to (5, 8400)
            x_center = predictions[0]
            y_center = predictions[1]
            width = predictions[2]
            height = predictions[3]
            confidence = predictions[4]
        else:
            cloudlog.warning(f"Unexpected YOLO output shape: {output.shape}")
            return detections

        # Apply confidence threshold
        cloudlog.debug(f"Confidence stats - min: {confidence.min():.4f}, max: {confidence.max():.4f}, mean: {confidence.mean():.4f}")
        cloudlog.debug(f"Coordinate ranges - x: [{x_center.min():.3f}, {x_center.max():.3f}], y: [{y_center.min():.3f}, {y_center.max():.3f}]")
        cloudlog.debug(f"Size ranges - w: [{width.min():.3f}, {width.max():.3f}], h: [{height.min():.3f}, {height.max():.3f}]")

        conf_mask = confidence >= conf_threshold
        num_above_threshold = np.sum(conf_mask)
        cloudlog.debug(f"Detections above threshold ({conf_threshold}): {num_above_threshold}")

        if not np.any(conf_mask):
            cloudlog.debug("No detections above confidence threshold")
            return detections

        # Use confidence mask (top-half filtering is now done by cropping input image)
        combined_mask = conf_mask

        # Filter detections by confidence
        x_center = x_center[combined_mask]
        y_center = y_center[combined_mask]
        width = width[combined_mask]
        height = height[combined_mask]
        confidence = confidence[combined_mask]

        # Convert from center format to corner format for NMS
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Stack boxes for NMS: [x1, y1, x2, y2, confidence]
        boxes = np.stack([x1, y1, x2, y2, confidence], axis=1)

        # Apply Non-Maximum Suppression
        keep_indices = self._apply_nms(boxes, nms_threshold)
        cloudlog.debug(f"After NMS: {len(keep_indices)} detections kept out of {len(boxes)}")

        # Convert back to normalized center format and add to detections
        large_detections_filtered = 0
        for idx in keep_indices:
            # Debug raw coordinates before normalization
            raw_x = x_center[idx]
            raw_y = y_center[idx]
            raw_w = width[idx]
            raw_h = height[idx]
            cloudlog.debug(f"Raw coords: x={raw_x:.3f}, y={raw_y:.3f}, w={raw_w:.3f}, h={raw_h:.3f}")

            # Check if coordinates are already normalized (0-1 range)
            if raw_x <= 1.0 and raw_y <= 1.0 and raw_w <= 1.0 and raw_h <= 1.0:
                # Coordinates are already normalized
                x_norm = raw_x
                y_norm = raw_y
                w_norm = raw_w
                h_norm = raw_h
                cloudlog.debug("Using raw coordinates (already normalized)")
            else:
                # Normalize pixel coordinates to [0, 1]
                x_norm = raw_x / MODEL_WIDTH
                y_norm = raw_y / MODEL_HEIGHT
                w_norm = raw_w / MODEL_WIDTH
                h_norm = raw_h / MODEL_HEIGHT
                cloudlog.debug(f"Normalizing: raw_x={raw_x:.1f}/{MODEL_WIDTH}={x_norm:.3f}")

            # Adjust coordinates to account for input cropping
            if TOP_HALF_FILTER_ENABLED and self._crop_params is not None:
                # Use the stored crop parameters from preprocessing
                orig_height = self._crop_params['original_height']
                orig_width = self._crop_params['original_width']
                start_y = self._crop_params['start_y']
                start_x = self._crop_params['start_x']
                cropped_height = self._crop_params['cropped_height']
                cropped_width = self._crop_params['cropped_width']

                # Map Y coordinates from the cropped region [0,1] back to the original image
                y_norm = y_norm * (cropped_height / orig_height) + (start_y / orig_height)

                # Map X coordinates from the cropped region [0,1] back to the original image
                x_norm = x_norm * (cropped_width / orig_width) + (start_x / orig_width)

                cloudlog.debug(f"Adjusted coordinates for square cropping: x={x_norm:.3f}, y={y_norm:.3f}")
            elif TOP_HALF_FILTER_ENABLED:
                # Fallback to original logic if crop parameters not available
                crop_height = CROP_END_PERCENT - CROP_START_PERCENT
                y_norm = y_norm * crop_height + CROP_START_PERCENT
                crop_width = crop_height  # Square aspect ratio
                x_norm = x_norm * crop_width + (1 - crop_width) / 2
                cloudlog.debug(f"Adjusted coordinates for square cropping (fallback): x={x_norm:.3f}, y={y_norm:.3f}")

            conf = confidence[idx]

            # Filter out detections that are too large (likely false positives)
            detection_area = w_norm * h_norm
            if detection_area > MAX_DETECTION_AREA:
                large_detections_filtered += 1
                cloudlog.debug(f"Filtered out large detection: area={detection_area:.3f} > {MAX_DETECTION_AREA:.3f}")
                continue

            detections.append([x_norm, y_norm, w_norm, h_norm, conf])
            cloudlog.debug(f"Final coords: x={x_norm:.3f}, y={y_norm:.3f}, w={w_norm:.3f}, h={h_norm:.3f}, conf={conf:.3f}, area={detection_area:.3f}")

        cloudlog.debug(f"Final detections: {len(detections)} (filtered out {large_detections_filtered} large detections)")

        # Debug: Visualize detections on the input image (only if DEBUG is enabled)
        if DEBUG_ENABLED and hasattr(self, '_debug_img_resized') and len(detections) > 0:
            debug_img = self._debug_img_resized.copy()
            for det in detections:
                x_norm, y_norm, w_norm, h_norm, conf = det
                # Convert normalized coordinates back to pixel coordinates for visualization
                x_pixel = int(x_norm * MODEL_WIDTH)
                y_pixel = int(y_norm * MODEL_HEIGHT)
                w_pixel = int(w_norm * MODEL_WIDTH)
                h_pixel = int(h_norm * MODEL_HEIGHT)

                # Adjust coordinates for visualization if we cropped the input
                if TOP_HALF_FILTER_ENABLED and self._crop_params is not None:
                    # The debug image is the cropped version, so we need to adjust coordinates
                    # back to the cropped image coordinate system for visualization
                    orig_height = self._crop_params['original_height']
                    orig_width = self._crop_params['original_width']
                    start_y = self._crop_params['start_y']
                    start_x = self._crop_params['start_x']
                    cropped_height = self._crop_params['cropped_height']
                    cropped_width = self._crop_params['cropped_width']

                    # Convert from original image coordinates back to cropped coordinates [0,1]
                    crop_y_norm = (y_norm - start_y / orig_height) / (cropped_height / orig_height)
                    crop_x_norm = (x_norm - start_x / orig_width) / (cropped_width / orig_width)

                    # Convert to pixel coordinates in the cropped image
                    y_pixel = int(crop_y_norm * MODEL_HEIGHT)
                    h_pixel = int(h_norm / (cropped_height / orig_height) * MODEL_HEIGHT)
                    x_pixel = int(crop_x_norm * MODEL_WIDTH)
                    w_pixel = int(w_norm / (cropped_width / orig_width) * MODEL_WIDTH)
                elif TOP_HALF_FILTER_ENABLED:
                    # Fallback to original logic for square cropping
                    crop_height = CROP_END_PERCENT - CROP_START_PERCENT
                    crop_width = crop_height  # Square aspect ratio

                    # Adjust Y coordinates
                    y_pixel = int((y_norm - CROP_START_PERCENT) / crop_height * MODEL_HEIGHT)
                    h_pixel = int(h_norm / crop_height * MODEL_HEIGHT)

                    # Adjust X coordinates
                    x_pixel = int((x_norm - (1 - crop_width) / 2) / crop_width * MODEL_WIDTH)
                    w_pixel = int(w_norm / crop_width * MODEL_WIDTH)

                # Draw bounding box
                cv2.rectangle(debug_img,
                             (x_pixel - w_pixel//2, y_pixel - h_pixel//2),
                             (x_pixel + w_pixel//2, y_pixel + h_pixel//2),
                             (0, 255, 0), 2)

                # Draw confidence
                cv2.putText(debug_img, f"{conf:.3f}",
                           (x_pixel - w_pixel//2, y_pixel - h_pixel//2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imwrite('/tmp/pothole_debug_detections.jpg', debug_img)
            cloudlog.info(f"Saved debug detections to /tmp/pothole_debug_detections.jpg with {len(detections)} detections")

        return detections

    def _apply_nms(self, boxes: np.ndarray, iou_threshold: float) -> list:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if len(boxes) == 0:
            return []

        # Sort by confidence (descending)
        scores = boxes[:, 4]
        sorted_indices = np.argsort(scores)[::-1]

        keep = []
        while len(sorted_indices) > 0:
            # Pick the box with highest confidence
            current = sorted_indices[0]
            keep.append(current)

            if len(sorted_indices) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[sorted_indices[1:]]

            ious = self._compute_iou(current_box, remaining_boxes)

            # Keep only boxes with IoU below threshold
            keep_mask = ious <= iou_threshold
            sorted_indices = sorted_indices[1:][keep_mask]

        return keep

    def _compute_iou(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute Intersection over Union (IoU) between one box and multiple boxes"""
        # box1: [x1, y1, x2, y2, conf]
        # boxes2: N x [x1, y1, x2, y2, conf]

        # Calculate intersection coordinates
        x1_max = np.maximum(box1[0], boxes2[:, 0])
        y1_max = np.maximum(box1[1], boxes2[:, 1])
        x2_min = np.minimum(box1[2], boxes2[:, 2])
        y2_min = np.minimum(box1[3], boxes2[:, 3])

        # Calculate intersection area
        intersection_width = np.maximum(0, x2_min - x1_max)
        intersection_height = np.maximum(0, y2_min - y1_max)
        intersection_area = intersection_width * intersection_height

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = box1_area + boxes2_area - intersection_area

        # Avoid division by zero
        union_area = np.maximum(union_area, 1e-6)

        return intersection_area / union_area

    def run(self, buf: VisionBuf) -> tuple[list, float]:
        """Run pothole detection on camera frame"""
        t1 = time.perf_counter()

        try:
            # Preprocess image
            img_input = self.preprocess_image(buf)

            # Convert to tinygrad tensor
            # Note: The model was compiled for CPU (NPY), so we need to use the same device
            # to avoid JIT compilation mismatches
            if TICI and not os.environ.get('USBGPU') and self.cl_ctx is not None:
                # Use OpenCL for Comma 3X
                img_tensor = qcom_tensor_from_opencl_address(
                    buf.mem_address,
                    (1, 3, MODEL_HEIGHT, MODEL_WIDTH),
                    dtype=dtypes.uint8
                )
            else:
                # Use appropriate device based on model type
                if os.environ.get('USBGPU') and MODEL_PKL_PATH_CUDA.exists():
                    # Use CUDA if USBGPU is set and CUDA model exists
                    img_tensor = Tensor(img_input, dtype=dtypes.float32, device='CUDA')
                else:
                    # Use CPU for desktop (model was compiled for NPY/CPU)
                    img_tensor = Tensor(img_input, dtype=dtypes.float32, device='NPY')

            # Run model inference
            model_output = self.model_run(images=img_tensor)

            # Convert output to numpy
            if hasattr(model_output, 'numpy'):
                output_np = model_output.numpy()
            else:
                output_np = model_output.data

            # Post-process detections
            detections = self.post_process_yolo(output_np, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

            # Debug logging
            if len(detections) > 0:
                cloudlog.info(f"Found {len(detections)} pothole detections: {detections}")
            else:
                # Log raw model output for debugging
                cloudlog.debug(f"Model output shape: {output_np.shape}, max: {output_np.max():.4f}, min: {output_np.min():.4f}")
                if output_np.shape[1] == 5:  # YOLO format
                    conf_scores = output_np[0, 4, :]  # Get confidence scores
                    max_conf = conf_scores.max()
                    conf_above_threshold = np.sum(conf_scores >= CONFIDENCE_THRESHOLD)
                    cloudlog.debug(f"Max confidence: {max_conf:.4f}, detections above threshold ({CONFIDENCE_THRESHOLD}): {conf_above_threshold}")

            # Add to temporal buffer for smoothing
            self.detection_buffer.append(detections)

            # Use latest detections (could implement temporal smoothing here)
            final_detections = detections

        except Exception as e:
            cloudlog.error(f"Pothole detection error: {e}")
            final_detections = []

        t2 = time.perf_counter()
        execution_time = t2 - t1

        return final_detections, execution_time


def main():
    config_realtime_process(7, 5)

    # Check if we're on TICI (Comma 3X) which has OpenCL support
    from openpilot.system.hardware import TICI
    if TICI:
        try:
            cl_context = CLContext()
            cloudlog.info("OpenCL context created successfully")
        except Exception as e:
            cloudlog.warning(f"Failed to create OpenCL context: {e}")
            cloudlog.warning("Running in CPU-only mode (performance may be reduced)")
            cl_context = None
    else:
        cloudlog.info("Non-TICI device detected, running in CPU-only mode")
        cl_context = None

    # Initialize model but don't load it yet
    model = None
    params = Params()
    cloudlog.info("potholed daemon starting (model will load when enabled)")

    # Connect to road camera
    vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True, cl_context)
    while not vipc_client.connect(False):
        time.sleep(0.1)
    cloudlog.warning(f"connected to road camera: {vipc_client.width}x{vipc_client.height}")

    pm = PubMaster(["potholeDetection"])

    frame_count = 0
    last_log_time = time.monotonic()
    last_param_check = time.monotonic()
    detection_enabled = False

    while True:
        buf = vipc_client.recv()
        if buf is None:
            continue

        # Check if pothole detection is enabled/disabled (every 1 second for responsiveness)
        current_time = time.monotonic()
        if current_time - last_param_check > 1.0:
            new_enabled = params.get_bool("PotholeDetectionEnabled")
            if new_enabled != detection_enabled:
                detection_enabled = new_enabled
                if detection_enabled:
                    if model is None:
                        cloudlog.info("Pothole detection enabled, loading model...")
                        model = ModelState(cl_context)
                        cloudlog.info("Pothole detection model loaded and ready")
                    else:
                        cloudlog.info("Pothole detection enabled")
                else:
                    cloudlog.info("Pothole detection disabled")
            last_param_check = current_time

        # Only process frames if detection is enabled and model is loaded
        if not detection_enabled or model is None:
            # Send empty message to keep the service alive
            msg = messaging.new_message('potholeDetection')
            msg.potholeDetection.frameId = vipc_client.frame_id
            msg.potholeDetection.modelExecutionTime = 0.0
            msg.potholeDetection.init('potholes', 0)
            pm.send('potholeDetection', msg)
            continue

        frame_count += 1

        # Run detection
        detections, execution_time = model.run(buf)

        # Debug: Log every 100 frames to show we're processing
        if frame_count % 100 == 0:
            cloudlog.info(f"Processing frame {frame_count}, execution time: {execution_time*1000:.1f}ms")

        # Create and send message
        msg = messaging.new_message('potholeDetection')
        msg.potholeDetection.frameId = vipc_client.frame_id
        msg.potholeDetection.modelExecutionTime = execution_time

        # Fill pothole data
        potholes = msg.potholeDetection.init('potholes', len(detections))
        for i, det in enumerate(detections):
            potholes[i].x = float(det[0])
            potholes[i].y = float(det[1])
            potholes[i].width = float(det[2])
            potholes[i].height = float(det[3])
            potholes[i].confidence = float(det[4])

            # Debug: Log coordinates being sent to UI
            cloudlog.info(f"UI coords: x={det[0]:.3f}, y={det[1]:.3f}, w={det[2]:.3f}, h={det[3]:.3f}, conf={det[4]:.3f}")

        pm.send('potholeDetection', msg)

        # Log performance every 5 seconds (only when detection is running)
        current_time = time.monotonic()
        if current_time - last_log_time > 5.0:
            if detection_enabled and model is not None:
                cloudlog.info(f"Pothole detection: {frame_count} frames, {execution_time*1000:.1f}ms avg, {len(detections)} detections")
            last_log_time = current_time
            frame_count = 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cloudlog.warning("got SIGINT")
    except Exception as e:
        cloudlog.error(f"potholed crashed: {e}")
        raise
