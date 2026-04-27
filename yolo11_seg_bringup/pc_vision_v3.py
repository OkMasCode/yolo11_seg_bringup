"""
ROS2 vision node for RGB + PointCloud segmentation, 3D centroid extraction,
CLIP embedding, and detection publishing.
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
import json
import threading
from time import perf_counter

from ultralytics import YOLO

from yolo11_seg_interfaces.msg import DetectedObjectV3, DetectedObjectV3Array
from .utils.siglip2_processor_2 import SIGLIPProcessor


# -------------------- CLASS ------------------- #

class VisionNode(Node):
    """
    ROS2 Node that handles Vision computation using camera's pointcloud
    """

    # ------------- Initialization ------------- #

    def __init__(self):
        # Node creation and startup log.
        super().__init__('vision_node')
        self.get_logger().info("Vision Node initialized\n")

        # ============= Parameters ============= #

        # Communication parameters
        self.declare_parameter('image_topic', '/jackal/sensors/camera_0/color/image') #/camera/camera/color/image_raw
        self.declare_parameter('depth_topic', '/jackal/sensors/camera_0/aligned_depth_to_color/image')
        self.declare_parameter('enable_visualization', True) # Flag for annotated image publisher
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.enable_vis = bool(self.get_parameter('enable_visualization').value)
        # YOLO parameters
        self.declare_parameter('model_path', '/home/workspace/yoloe-26m-seg.pt')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf', 0.45)
        self.declare_parameter('iou', 0.35)
        self.model_path = self.get_parameter('model_path').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        # CLIP parameters
        self.declare_parameter('CLIP_model_name', '/home/workspace/local_siglip_model')
        self.declare_parameter('CLIP_model_path', '/home/workspace/siglip_vision_pooled_384_fp16.engine') 
        self.declare_parameter('robot_command_file', '/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json')
        self.declare_parameter('prompt_check_interval', 5.0)
        self.declare_parameter('masked_score_weight', 0.85)
        self.declare_parameter('unmasked_score_weight', 0.15)
        self.declare_parameter('enable_paper_capture', False)
        self.declare_parameter('paper_capture_class', 'bed')
        self.declare_parameter('paper_images_output_dir', '/home/workspace/ros2_ws/src/yolo11_seg_bringup/images')
        self.declare_parameter('annotated_font_size', 0.6)
        self.declare_parameter('annotated_line_width', 1)
        self.CLIP_model_name = self.get_parameter('CLIP_model_name').value
        self.CLIP_model_path = self.get_parameter('CLIP_model_path').value
        self.robot_command_file = self.get_parameter('robot_command_file').value
        self.prompt_check_interval = float(self.get_parameter('prompt_check_interval').value)
        self.masked_score_weight = float(self.get_parameter('masked_score_weight').value)
        self.unmasked_score_weight = float(self.get_parameter('unmasked_score_weight').value)
        self.enable_paper_capture = bool(self.get_parameter('enable_paper_capture').value)
        self.paper_capture_class = str(self.get_parameter('paper_capture_class').value).strip()
        self.paper_images_output_dir = str(self.get_parameter('paper_images_output_dir').value).strip()
        self.annotated_font_size = float(self.get_parameter('annotated_font_size').value)
        self.annotated_line_width = max(1, int(self.get_parameter('annotated_line_width').value))

        # =========== Internal Topics / Runtime =========== #

        self.anno_topic = '/vision/annotated_image'
        self.detection_topic = '/vision/detections'
        self.text_emb_publish_topic = '/vision/text_embedding'
        self.frame_skip = 5
        self.CLASS_NAMES = ["person", "bus", "tree"]        
        goal_class = self._read_goal_from_command_file()
        # If a valid goal class is found in the command file, ensure it's included in CLASS_NAMES for detection.
        if goal_class:
            if goal_class in self.CLASS_NAMES:
                self.get_logger().info(
                    f"Goal class '{goal_class}' already exists in CLASS_NAMES. Using existing class list."
                )
            else:
                self.CLASS_NAMES.append(goal_class)
                self.get_logger().info(
                    f"Goal class '{goal_class}' not found in CLASS_NAMES. Appended to class list."
                )
        else:
            self.get_logger().warn(
                "No valid goal found in robot_command.json. Using default CLASS_NAMES."
            )
        # Load YOLO model
        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path, task='segment')
        self.model.set_classes(self.CLASS_NAMES)
        self.get_logger().info(f"YOLO classes set to: {self.CLASS_NAMES}")
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on device: {self.device}\n")
        self.clip = SIGLIPProcessor(
            engine_path=self.CLIP_model_path,
            model_name=self.CLIP_model_name,
            masked_score_weight=self.masked_score_weight,
            unmasked_score_weight=self.unmasked_score_weight,
        )
        self.get_logger().info(
            "CLIP blend weights normalized to: "
            f"masked={self.clip.masked_score_weight:.2f}, "
            f"unmasked={self.clip.unmasked_score_weight:.2f}"
        )
        # CV bridge and camera intrinsics placeholders.
        self.bridge = CvBridge()
        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, 
            self.image_topic, 
            self.rgb_callback,
            qos_profile=qos_sensor,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile=qos_sensor,
        )
        self.get_logger().info(f"Subscribed to: {self.image_topic}")
        self.get_logger().info(f"Subscribed to depth: {self.depth_topic}")
        # Publishers    
        self.detections_pub = self.create_publisher(DetectedObjectV3Array, self.detection_topic, 10)
        self.text_emb_pub = self.create_publisher(Float32MultiArray, self.text_emb_publish_topic, 10)
        if self.enable_vis:
            self.vis_pub = self.create_publisher(Image, self.anno_topic, qos_profile=qos_sensor)
        # State Variables
        self.frame_count = 0
        self.class_colors = {}
        self.current_clip_prompt = None
        self.goal_text_embedding = None
        self.clip_state_lock = threading.Lock()
        self.depth_state_lock = threading.Lock()
        self.latest_depth_msg = None
        self.paper_capture_interval_sec = 5.0
        self.last_paper_capture_time = -self.paper_capture_interval_sec
        if self.enable_paper_capture:
            os.makedirs(self.paper_images_output_dir, exist_ok=True)
            self.get_logger().info(
                "Paper image capture configured: "
                f"class='{self.paper_capture_class}', interval={self.paper_capture_interval_sec:.1f}s, "
                f"dir='{self.paper_images_output_dir}'"
            )
        else:
            self.get_logger().info("Paper image capture is disabled by parameter 'enable_paper_capture'.")
        # Timing instrumentation
        self.timing_stats = {
            'yolo_inference': [],
            'detections_processing': [],
            'clip_encoding': [],
            'publishing': [],
            'total_frame': []
        }
        self.timing_window = 30 # (Frames)
        # Initialize the prompt immediately on startup
        self._load_clip_prompt()
        self.command_timer = self.create_timer(self.prompt_check_interval, self._timer_publish_embedding)
        self.get_logger().info("Vision Node Ready.")

    # ---------------- Callbacks --------------- #

    def rgb_callback(self, rgb_msg: Image):
        """
        RGB callback.
        """
        input_stamp = rgb_msg.header.stamp
        self.get_logger().info(
            f"[rgb_callback] input_rgb_stamp={input_stamp.sec}.{input_stamp.nanosec:09d} "
            f"frame_id='{rgb_msg.header.frame_id}' size={rgb_msg.width}x{rgb_msg.height}",
            throttle_duration_sec=2.0,
        )
        self.process_frame(rgb_msg)

    def depth_callback(self, depth_msg: Image):
        """Cache the most recent aligned depth frame for paper-capture exports."""
        with self.depth_state_lock:
            self.latest_depth_msg = depth_msg
        self.get_logger().debug(
            f"[depth_callback] stamp={depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec:09d} "
            f"frame_id='{depth_msg.header.frame_id}' encoding='{depth_msg.encoding}' size={depth_msg.width}x{depth_msg.height}"
        )
   
    # --------------- Main Methods ------------- #

    def process_frame(self, rgb_msg: Image):
        """
        Process a single RGB frame.
        """
        self.frame_count += 1
        frame_start = perf_counter()
        try:
            # Convert ROS Image to OpenCV BGR frame.
            cv_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            height, width = cv_bgr.shape[:2]

            # ======== YOLO inference ======== #

            yolo_start = perf_counter()
            results = self.model.track(
                source=cv_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                retina_masks=True,
                stream=False,
                verbose=False,
                persist=True,
                tracker="botsort.yaml",
            )
            yolo_time = (perf_counter() - yolo_start) * 1000
            self.timing_stats['yolo_inference'].append(yolo_time)
            res = results[0]
            if not hasattr(res, 'masks') or len(res.boxes) == 0:
                self.get_logger().info("No detections in this frame", throttle_duration_sec=2.0)
                return
            
            # ======== process results ======== #

            self._process_detections(res, cv_bgr, rgb_msg, height, width)
            if self.enable_vis:
                # result.plot() draws boxes, masks, and IDs on the image
                annotated_frame = res.plot(
                    font_size=self.annotated_font_size,
                    line_width=self.annotated_line_width,
                )
                vis_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                vis_msg.header = rgb_msg.header
                self.vis_pub.publish(vis_msg)
                self.get_logger().info(f"Published annotated image (frame {self.frame_count})", throttle_duration_sec=2.0)
            # Record end-to-end latency for successful frames.
            total_time = (perf_counter() - frame_start) * 1000
            self.timing_stats['total_frame'].append(total_time)
            # Print timing stats every N frames
            if self.frame_count % self.timing_window == 0:
                self._print_timing_stats()
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def _process_detections(self, res, cv_bgr, rgb_msg, height, width):
        """ 
        Unified pipeline
        """

        # ===== GPU-to-CPU transfer ===== #

        # Move detection tensors from GPU to CPU NumPy arrays for geometric processing.
        gpu_transfer_start = perf_counter()
        xyxy = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.zeros(len(clss), dtype=float)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.arange(len(clss))
        names = np.array([res.names[c] for c in clss])
        # Transfer all masks at once
        if hasattr(res, 'masks') and res.masks is not None:
            masks_np = res.masks.data.detach().cpu().numpy()
        else:
            masks_np = None
        gpu_transfer_time = (perf_counter() - gpu_transfer_start) * 1000
        self.get_logger().debug(f"GPU-to-CPU transfer: {gpu_transfer_time:.2f}ms")
        # Per-frame containers: all detections and CLIP candidates for batch inference.
        frame_detections = []
        batch_queue = []
        # Run CLIP every N frames to reduce compute load.
        do_clip_frame = (self.frame_count % max(1, self.frame_skip) == 0)
        capture_due = self._is_paper_capture_due()

        # ===== detection processing ===== #

        # Process each instance: mask filtering, crop prep.
        det_process_start = perf_counter()
        # Iterate through detections to build a unified data structure for downstream processing and publishing.
        for i in range(len(xyxy)):
            result = self.process_single_detection(
                i, xyxy[i], clss[i], names[i], ids[i], confs[i], masks_np,
                cv_bgr,
                height, width, do_clip_frame, capture_due
            )
            if result is None:
                continue
            det_entry = result
            frame_detections.append(det_entry)
            if det_entry.get("masked_crop") is not None and det_entry.get("unmasked_crop") is not None:
                batch_queue.append(det_entry)
        det_proc_time = (perf_counter() - det_process_start) * 1000
        self.timing_stats['detections_processing'].append(det_proc_time)
        self.get_logger().debug(f"Detection processing ({len(frame_detections)} detections): {det_proc_time:.2f}ms")

        if capture_due:
            self._maybe_save_paper_images(frame_detections, cv_bgr, res)
            
        # ===== SigLIP encoding ===== #

        # Batch SigLIP image encoding for all prepared crops.
        if batch_queue:
            try:
                clip_start = perf_counter()
                masked_images = [item["masked_crop"] for item in batch_queue]
                unmasked_images = [item["unmasked_crop"] for item in batch_queue]
                all_images = masked_images + unmasked_images
                all_embeddings = self.clip.encode_images_batch(all_images)
                half_idx = len(masked_images)
                masked_embeddings = all_embeddings[:half_idx]
                unmasked_embeddings = all_embeddings[half_idx:]
                pair_count = min(len(masked_embeddings), len(unmasked_embeddings), len(batch_queue))
                for i in range(pair_count):
                    batch_queue[i]["masked_embedding"] = masked_embeddings[i]
                    batch_queue[i]["unmasked_embedding"] = unmasked_embeddings[i]
                    batch_queue[i]["masked_crop"] = None
                    batch_queue[i]["unmasked_crop"] = None
                clip_time = (perf_counter() - clip_start) * 1000
                self.timing_stats['clip_encoding'].append(clip_time)
                self.get_logger().info(
                    f"SigLIP encoding completed: paired={pair_count}/{len(batch_queue)} crops in {clip_time:.2f}ms",
                    throttle_duration_sec=2.0,
                )
            except Exception as e:
                self.get_logger().error(f"Batch SigLIP inference failed: {e}")

        # ======== Publishing ========= #
        
        # Publish structured detections and visualization markers.
        pub_start = perf_counter()
        self.publish_custom_detections(frame_detections, rgb_msg.header)
        pub_time = (perf_counter() - pub_start) * 1000
        self.timing_stats['publishing'].append(pub_time)
        self.get_logger().debug(f"Publishing: {pub_time:.2f}ms")

    def process_single_detection(self, idx, bbox, class_id, class_name, instance_id, confidence, masks_np,
                                 cv_bgr,
                                 height, width, do_clip_frame, capture_due):
        """ 
        prepare data container. 
        Returns detection_dictionary
        """

        # Clamp bounding box to image boundaries.
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width - 1))
        y2 = max(0, min(int(y2), height - 1))
        # Reject invalid boxes and missing masks.
        if x2 <= x1 or y2 <= y1:
            return None
        if masks_np is None or idx >= masks_np.shape[0]:
            return None
        # Get mask for this instance (already on CPU).
        m = masks_np[idx]
        if m.shape[0] != height or m.shape[1] != width:
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        # Convert to standard uint8 mask (0 or 255) for OpenCV
        mask_uint8 = (m > 0.5).astype(np.uint8) * 255
        # Erode the mask to prevent depth bleeding.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=3)
        detection_entry = {
            "class_id": int(class_id),
            "instance_id": int(instance_id),
            "object_name": class_name,
            "confidence": float(confidence),
            "bbox": (x1, y1, x2, y2),
            "mask_uint8": mask_uint8,
            "embedding": None,
            "masked_embedding": None,
            "unmasked_embedding": None,
            "masked_crop": None,
            "unmasked_crop": None,
            "mask_ros": eroded_mask
        }
        # Prepare SigLIP crop (if current frame is scheduled for SigLIP).
        should_prepare_crop = do_clip_frame or (capture_due and class_name == self.paper_capture_class)
        if should_prepare_crop:
            masked_crop, unmasked_crop = self.clip.prepare_crops(
                cv_bgr,
                mask_uint8,
                (x1, y1, x2, y2),
            )
            if masked_crop is not None and unmasked_crop is not None:
                detection_entry["masked_crop"] = masked_crop
                detection_entry["unmasked_crop"] = unmasked_crop
        return detection_entry

    def _is_paper_capture_due(self):
        """Return True when a new paper snapshot should be taken."""
        if not self.enable_paper_capture:
            return False
        now = perf_counter()
        return (now - self.last_paper_capture_time) >= self.paper_capture_interval_sec

    def _maybe_save_paper_images(self, frame_detections, cv_bgr, res):
        """Save requested intermediate images for one target-class detection every interval."""
        if not self.enable_paper_capture:
            return
        target_class = self.paper_capture_class
        if not target_class:
            self.get_logger().warn("paper_capture_class is empty; skipping paper image capture.")
            return

        # Always save full-frame views on interval.
        annotated_image = res.plot(
            font_size=self.annotated_font_size,
            line_width=self.annotated_line_width,
        )
        saved = []
        if self._write_debug_image("raw_image_cv_bgr.png", cv_bgr):
            saved.append("raw")
        if self._write_debug_image("yolo_overview_annotated_image.png", annotated_image):
            saved.append("annotated_all_detections")
        depth_saved = self._save_latest_depth_images()
        saved.extend(depth_saved)

        matching = [
            det for det in frame_detections
            if str(det.get("object_name", "")).strip().lower() == target_class.lower()
        ]
        if not matching:
            self.get_logger().info(
                f"Paper capture due: saved {saved}. No detection for class '{target_class}' in this frame.",
                throttle_duration_sec=2.0,
            )
            self.last_paper_capture_time = perf_counter()
            return

        selected = max(matching, key=lambda item: float(item.get("confidence", 0.0)))
        mask_uint8 = selected.get("mask_uint8")
        eroded_mask = selected.get("mask_ros")
        masked_crop = selected.get("masked_crop")
        unmasked_crop = selected.get("unmasked_crop")

        # Ensure crops exist even when this frame is not part of CLIP cadence.
        if masked_crop is None or unmasked_crop is None:
            bbox = selected.get("bbox")
            if mask_uint8 is not None and bbox is not None:
                masked_crop, unmasked_crop = self.clip.prepare_crops(cv_bgr, mask_uint8, bbox)

        if mask_uint8 is not None and self._write_debug_image("binary_mask_mask_uint8.png", mask_uint8):
            saved.append("mask")
        if eroded_mask is not None and self._write_debug_image("eroded_mask.png", eroded_mask):
            saved.append("eroded")

        unmasked_crop_bgr = self._to_bgr_image(unmasked_crop)
        if unmasked_crop_bgr is not None and self._write_debug_image("unmasked_crop.png", unmasked_crop_bgr):
            saved.append("unmasked_crop")

        masked_crop_bgr = self._to_bgr_image(masked_crop)
        if masked_crop_bgr is not None and self._write_debug_image("masked_crop.png", masked_crop_bgr):
            saved.append("masked_crop")

        self.last_paper_capture_time = perf_counter()
        self.get_logger().info(
            f"Saved paper images for class '{target_class}' (instance_id={selected.get('instance_id')}) "
            f"to '{self.paper_images_output_dir}'. Files: {saved}"
        )

    def _save_latest_depth_images(self):
        """Save latest aligned depth frame in raw and visualized forms for paper captures."""
        with self.depth_state_lock:
            depth_msg = self.latest_depth_msg
        if depth_msg is None:
            self.get_logger().warn(
                "Paper capture requested but no depth frame received yet.",
                throttle_duration_sec=2.0,
            )
            return []
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Failed converting depth image for paper capture: {e}")
            return []
        if depth_img is None:
            return []
        saved = []
        depth_mm_u16 = self._depth_to_uint16_mm(depth_img)
        depth_mm_u16_for_export = self._brighten_depth_uint16_for_export(depth_mm_u16)
        if depth_mm_u16_for_export is not None and self._write_debug_image("aligned_depth_raw_mm_u16.png", depth_mm_u16_for_export):
            saved.append("depth_raw_mm_u16")
        depth_vis = self._depth_to_visualization(depth_img)
        if depth_vis is not None and self._write_debug_image("aligned_depth_visualization.png", depth_vis):
            saved.append("depth_visualization")
        return saved

    def _depth_to_uint16_mm(self, depth_img):
        """Convert depth image to uint16 millimeters for lossless-ish PNG export."""
        try:
            if depth_img.dtype == np.uint16:
                return depth_img
            depth = depth_img.astype(np.float32, copy=False)
            invalid = ~np.isfinite(depth)
            depth = np.where(invalid | (depth <= 0.0), 0.0, depth)
            depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
            return depth_mm
        except Exception as e:
            self.get_logger().warn(f"Failed converting depth to uint16 mm: {e}")
            return None

    def _depth_to_visualization(self, depth_img):
        """Create an 8-bit depth visualization for quick inspection."""
        try:
            depth = depth_img.astype(np.float32, copy=False)
            valid = np.isfinite(depth) & (depth > 0.0)
            if not np.any(valid):
                return None
            vmin = np.percentile(depth[valid], 5)
            vmax = np.percentile(depth[valid], 95)
            if vmax <= vmin:
                vmax = vmin + 1e-3
            depth_norm = np.zeros(depth.shape, dtype=np.float32)
            depth_norm[valid] = (depth[valid] - vmin) / (vmax - vmin)
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
            depth_u8 = (depth_norm * 255.0).astype(np.uint8)
            return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        except Exception as e:
            self.get_logger().warn(f"Failed generating depth visualization: {e}")
            return None

    def _brighten_depth_uint16_for_export(self, depth_mm_u16):
        """Stretch uint16 depth range for brighter PNG viewing."""
        if depth_mm_u16 is None:
            return None
        try:
            valid = depth_mm_u16 > 0
            if not np.any(valid):
                return depth_mm_u16
            p95 = float(np.percentile(depth_mm_u16[valid], 95))
            if p95 <= 0.0:
                return depth_mm_u16
            target = 55000.0
            gain = target / p95
            gain = float(np.clip(gain, 1.0, 20.0))
            bright = np.clip(depth_mm_u16.astype(np.float32) * gain, 0.0, 65535.0).astype(np.uint16)
            self.get_logger().debug(f"Depth export brightness gain applied: {gain:.2f}")
            return bright
        except Exception as e:
            self.get_logger().warn(f"Failed brightening uint16 depth export: {e}")
            return depth_mm_u16

    def _write_debug_image(self, filename, image):
        """Write image to output directory and overwrite existing file."""
        if image is None:
            return False
        path = os.path.join(self.paper_images_output_dir, filename)
        try:
            ok = cv2.imwrite(path, image)
            if not ok:
                self.get_logger().warn(f"cv2.imwrite failed for '{path}'")
            return ok
        except Exception as e:
            self.get_logger().error(f"Failed to save image '{path}': {e}")
            return False

    def _to_bgr_image(self, image):
        """Convert common crop formats (NumPy/PIL) to OpenCV BGR image."""
        if image is None:
            return None
        try:
            if isinstance(image, np.ndarray):
                arr = image
            else:
                # Handles PIL images or similar array-compatible objects.
                arr = np.array(image)
            if arr.ndim == 2:
                return arr
            if arr.ndim != 3:
                return None
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().warn(f"Could not convert crop image for saving: {e}")
            return None
    
    # ----------- Secondary Methods ------------ #

    def _load_clip_prompt(self):
        """
        Load SigLIP prompt from robot_command.json and regenerate embedding
        only when prompt content changes.
        """
        if not os.path.exists(self.robot_command_file):
            self.get_logger().warn(
                f"robot_command file not found: '{self.robot_command_file}'. SigLIP scoring is paused.",
                throttle_duration_sec=10.0,
            )
            return
        try:
            with open(self.robot_command_file, 'r') as f:
                data = json.load(f)
            # Required source: robot_command.json -> clip_prompts
            # Current format uses a single prompt string.
            clip_prompt = data.get('clip_prompts', None)
            if isinstance(clip_prompt, str):
                prompt_value = clip_prompt.strip()
            else:
                prompt_value = ''
            prompt_texts = None
            prompt_key = None
            if prompt_value:
                # Encode the single clip prompt string into a list
                prompt_texts = self.clip.build_prompt_list(prompt_value)
                prompt_key = prompt_value
            if not prompt_texts:
                self.get_logger().warn(
                    f"No valid 'clip_prompts' value in '{self.robot_command_file}'.",
                    throttle_duration_sec=5.0,
                )
                return
            if prompt_key == self.current_clip_prompt:
                return
            new_embedding = self.clip.encode_text(prompt_texts)
            if new_embedding is None:
                self.get_logger().warn("Prompt encoding returned no embedding", throttle_duration_sec=5.0)
                return
            with self.clip_state_lock:
                self.current_clip_prompt = prompt_key
                self.goal_text_embedding = new_embedding
            self.get_logger().info(
                f"Updated SigLIP prompt from robot_command clip_prompts: '{prompt_key}'",
                throttle_duration_sec=2.0,
            )
        except Exception as e:
            self.get_logger().error(f"Error loading robot_command siglip prompt: {e}")

    def _timer_publish_embedding(self):
        self._load_clip_prompt()
        with self.clip_state_lock:
            if self.goal_text_embedding is not None:
                msg = Float32MultiArray()
                # Flatten the text embedding into a list
                data_list = [float(val) for val in self.goal_text_embedding.flatten()]
                # Append the model's specific scale and bias to the end of the array
                data_list.append(self.clip.cached_logit_scale)
                data_list.append(self.clip.cached_logit_bias)
                msg.data = data_list
                self.text_emb_pub.publish(msg)

    def _read_goal_from_command_file(self):
        """Read robot command JSON and return the goal string, if available."""
        try:
            if not os.path.exists(self.robot_command_file):
                return None
            with open(self.robot_command_file, 'r') as f:
                data = json.load(f)

            goal = data.get('goal', None)
            if isinstance(goal, str):
                goal = goal.strip()
            return goal if goal else None
        except Exception as e:
            self.get_logger().error(f"Error reading goal from robot_command.json: {e}")
            return None
    
    def _print_timing_stats(self):
        """
        Print aggregated timing statistics for the last N frames.
        """
        if not self.timing_stats['total_frame']:
            return
        def get_stats(timings):
            if not timings:
                return 0, 0, 0
            return min(timings), np.mean(timings), max(timings)
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info(f"TIMING STATS (frames {self.frame_count - self.timing_window + 1}-{self.frame_count})")
        self.get_logger().info("="*70)
        for stat_name, timings in self.timing_stats.items():
            if timings:
                min_t, avg_t, max_t = get_stats(timings)
                self.get_logger().info(f"{stat_name:25s}: min={min_t:6.2f}ms | avg={avg_t:6.2f}ms | max={max_t:6.2f}ms")
        self.get_logger().info("="*70 + "\n")
        # Reset timing stats
        for key in self.timing_stats:
            self.timing_stats[key] = []
    
    # ---------------- Publishers -------------- #

    def publish_custom_detections(self, detections, header):
        """ 
        Iterate through the unified list, package into an Array, and publish once. 
        """
        if not detections:
            return
        array_msg = DetectedObjectV3Array()
        array_msg.header.stamp = header.stamp
        array_msg.header.frame_id = header.frame_id
        self.get_logger().info(
            f"[publish_detections] output_stamp={array_msg.header.stamp.sec}.{array_msg.header.stamp.nanosec:09d} "
            f"frame_id='{array_msg.header.frame_id}' count={len(detections)}",
            throttle_duration_sec=2.0,
        )
        published_rows = []
        sim_comp_print = False
        for det in detections:
            msg = DetectedObjectV3() 
            # Basic Info
            msg.class_name = det["object_name"]
            msg.instance_id = int(det["instance_id"]) 
            msg.confidence = float(det.get("confidence", 0.0))
            # Embeddings
            current_emb = det.get("embedding")
            masked_emb = det.get("masked_embedding")
            unmasked_emb = det.get("unmasked_embedding")
            with self.clip_state_lock:
                text_embedding = None if self.goal_text_embedding is None else self.goal_text_embedding.copy()
            # Change these to match the .msg file exactly!
            msg.image_embedding_masked = masked_emb.tolist() if masked_emb is not None else []
            msg.image_embedding_unmasked = unmasked_emb.tolist() if unmasked_emb is not None else []
            msg.text_embedding = text_embedding.tolist() if text_embedding is not None else []
            # Compute goal similarity
            prob_goal = 0.0
            sim_comp = False
            if masked_emb is not None and unmasked_emb is not None and text_embedding is not None:
                prob_goal = self.clip.compute_blended_match_score(masked_emb, unmasked_emb, text_embedding)
                msg.similarity = float(prob_goal)
                sim_comp = True
            elif current_emb is not None and text_embedding is not None:
                # Fallback for detections that only have one embedding.
                prob_goal = self.clip.compute_match_score(current_emb, text_embedding)
                msg.similarity = float(prob_goal)
                sim_comp = True
            else:
                msg.similarity = 0.0
            if sim_comp:
                sim_comp_print = True
            published_rows.append({
                "instance_id": int(det["instance_id"]),
                "class_name": det["object_name"],
                "confidence": float(det.get("confidence", 0.0)),
                "score": float(msg.similarity),
            })
            if "mask_ros" in det and det["mask_ros"] is not None:
                # Convert the uint8 NumPy array to a ROS Image message.
                # 'mono8' is the standard encoding for single-channel grayscale/binary masks.
                msg.mask = self.bridge.cv2_to_imgmsg(det["mask_ros"], encoding="mono8")
            else:
                self.get_logger().warning(f"Detection {det['instance_id']} is missing 'mask_ros'")
            # Append to array
            array_msg.detections.append(msg)
        # Publish the entire frame of detections simultaneously
        self.detections_pub.publish(array_msg)
        if sim_comp_print:
            self._log_published_detections_table(published_rows, array_msg.header)

    def _log_published_detections_table(self, rows, header):
        """Log a compact table of the detections that were actually published."""
        if not rows:
            return
        max_confidence = max((row["confidence"] for row in rows), default=0.0)
        max_score = max((row["score"] for row in rows), default=0.0)
        table_rows = []
        for index, row in enumerate(rows, start=1):
            rel_confidence = (row["confidence"] / max_confidence) if max_confidence > 0.0 else 0.0
            rel_score = (row["score"] / max_score) if max_score > 0.0 else 0.0
            table_rows.append([
                str(index),
                str(row["instance_id"]),
                row["class_name"],
                f"{row['confidence']:.3f}",
                f"{rel_confidence:.2f}",
                f"{row['score']:.3f}",
                f"{rel_score:.2f}",
            ])
        headers = ["idx", "id", "class", "conf", "rel_conf", "score", "rel_score"]
        widths = [len(title) for title in headers]
        for row in table_rows:
            for column_index, value in enumerate(row):
                widths[column_index] = max(widths[column_index], len(value))
        def format_row(values):
            return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"
        separator = "| " + " | ".join("-" * width for width in widths) + " |"
        lines = [
            f"[publish_detections] frame={self.frame_count} stamp={header.stamp.sec}.{header.stamp.nanosec:09d} frame_id='{header.frame_id}'",
            format_row(headers),
            separator,
        ]
        lines.extend(format_row(row) for row in table_rows)
        self.get_logger().info("\n" + "\n".join(lines))

# -------------------- MAIN -------------------- # 

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
