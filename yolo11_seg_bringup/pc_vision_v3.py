"""
ROS2 vision node for RGB + PointCloud segmentation, 3D centroid extraction,
CLIP embedding, and detection publishing.
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
import json
from time import perf_counter

from ultralytics import YOLO

from yolo11_seg_interfaces.msg import DetectedObjectV3, DetectedObjectV3Array
from .utils.clip_processor import CLIPProcessor


# -------------------- CLASS ------------------- #

class NoPCVisionNode(Node):
    """
    ROS2 Node that handles Vision computation using camera's pointcloud
    """

    # ------------- Initialization ------------- #

    def __init__(self):
        # Node creation and startup log.
        super().__init__('no_pc_vision_node')
        self.get_logger().info("NoPCVisionNode initialized\n")

        # ============= Parameters ============= #

        # Communication parameters
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw') #/camera/color/image_raw
        self.declare_parameter('enable_visualization', True)

        self.image_topic = self.get_parameter('image_topic').value
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
        self.declare_parameter('CLIP_model_name', 'ViT-B-16-SigLIP')
        self.declare_parameter('robot_command_file', '/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json')
        self.declare_parameter('map_file_path', '/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/map.json')
        self.declare_parameter('square_crop_scale', 1.2)

        self.CLIP_model_name = self.get_parameter('CLIP_model_name').value
        self.robot_command_file = self.get_parameter('robot_command_file').value
        self.square_crop_scale = float(self.get_parameter('square_crop_scale').value)

        # =========== Internal Topics / Runtime =========== #

        self.anno_topic = '/vision/annotated_image'
        self.detection_topic = '/vision/detections'

        self.frame_skip = 5
        self.prompt_check_interval = 760.0  # Check for new prompts every 760 seconds

        self.CLASS_NAMES = ["chair", "bottle", "keyboard", "mouse", "tv"]

        goal_class = self._read_goal_from_command_file()
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

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path, task='segment')
        self.model.set_classes(self.CLASS_NAMES)
        self.get_logger().info(f"YOLO classes set to: {self.CLASS_NAMES}")

        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on device: {self.device}\n")
        self.clip = CLIPProcessor(
            device=self.device, 
            model_name=self.CLIP_model_name, 
            pretrained="webli"
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
        
        self.get_logger().info(f"Subscribed to: {self.image_topic}")

        # Publishers    
        self.detections_pub = self.create_publisher(DetectedObjectV3Array, self.detection_topic, 10)
        if self.enable_vis:
            self.vis_pub = self.create_publisher(Image, self.anno_topic, 10)

        # State Variables
        self.frame_count = 0
        self.class_colors = {}
        self.current_clip_prompt = None
        self.goal_text_embedding = None

        # Timing instrumentation
        self.timing_stats = {
            'yolo_inference': [],
            'detections_processing': [],
            'clip_encoding': [],
            'publishing': [],
            'total_frame': []
        }
        self.timing_window = 30  # Print stats every N frames

        self._load_clip_prompt()
        self.command_timer = self.create_timer(self.prompt_check_interval, self._load_clip_prompt)

        self.get_logger().info("Vision Node Ready.")

    # ---------------- Callbacks --------------- #

    def rgb_callback(self, rgb_msg: Image):
        """
        RGB callback.
        """

        self.process_frame(rgb_msg)
   
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

            # ===== YOLO INFERENCE TIMING =====
            yolo_start = perf_counter()
            # Run YOLO tracking + segmentation on current frame.
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
            
            # Run geometry extraction, CLIP embedding, and message publishing.
            self._process_detections(res, cv_bgr, rgb_msg, height, width)

            if self.enable_vis:
                # result.plot() draws boxes, masks, and IDs on the image
                annotated_frame = res.plot()
                vis_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                vis_msg.header = rgb_msg.header
                self.vis_pub.publish(vis_msg)
                self.get_logger().info(f"Published annotated image (frame {self.frame_count})", throttle_duration_sec=2.0)
            
            # ===== TOTAL FRAME TIMING =====
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
        # ===== GPU-to-CPU TRANSFER TIMING =====
        # Move detection tensors from GPU to CPU NumPy arrays for geometric processing.
        gpu_transfer_start = perf_counter()
        xyxy = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.zeros(len(clss), dtype=float)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.arange(len(clss))
        
        names = np.array([res.names[c] for c in clss])
        # Transfer all masks at once (faster than per-detection transfer).
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

        # ===== DETECTION PROCESSING TIMING =====
        # Process each instance: mask filtering, 3D centroid, crop prep.
        det_process_start = perf_counter()
        for i in range(len(xyxy)):
            result = self.process_single_detection(
                i, xyxy[i], clss[i], names[i], ids[i], confs[i], masks_np,
                cv_bgr,
                height, width, do_clip_frame
            )

            if result is None:
                continue

            det_entry = result
            frame_detections.append(det_entry)
            if det_entry.get("crop") is not None:
                batch_queue.append(det_entry)
        
        det_proc_time = (perf_counter() - det_process_start) * 1000
        self.timing_stats['detections_processing'].append(det_proc_time)
        self.get_logger().debug(f"Detection processing ({len(frame_detections)} detections): {det_proc_time:.2f}ms")
            
        # ===== CLIP ENCODING TIMING =====
        # Batch CLIP image encoding for all prepared crops.
        if batch_queue:
            try:
                clip_start = perf_counter()
                # Extract just the images for the batch
                images = [item["crop"] for item in batch_queue]
                
                # Run inference on all at once
                embeddings = self.clip.encode_images_batch(images)
                
                # Re-associate embeddings with metadata and release image crops.
                for i, emb in enumerate(embeddings):
                    batch_queue[i]["embedding"] = emb
                    # Clear the image to free memory
                    batch_queue[i]["crop"] = None
                
                clip_time = (perf_counter() - clip_start) * 1000
                self.timing_stats['clip_encoding'].append(clip_time)
                self.get_logger().debug(f"CLIP encoding ({len(batch_queue)} crops): {clip_time:.2f}ms")
            except Exception as e:
                self.get_logger().error(f"Batch CLIP inference failed: {e}")

        # ===== PUBLISHING TIMING =====
        # Publish structured detections and visualization markers.
        pub_start = perf_counter()
        self.publish_custom_detections(frame_detections, rgb_msg.header)
        
        pub_time = (perf_counter() - pub_start) * 1000
        self.timing_stats['publishing'].append(pub_time)
        self.get_logger().debug(f"Publishing: {pub_time:.2f}ms")

    def process_single_detection(self, idx, bbox, class_id, class_name, instance_id, confidence, masks_np,
                                 cv_bgr,
                                 height, width, do_clip_frame):
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
        
        # 1. Convert to standard uint8 mask (0 or 255) for OpenCV
        mask_uint8 = (m > 0.5).astype(np.uint8) * 255
        
        # 2. SOTA Edge Fix: Erode the mask to prevent depth bleeding.
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
        
        # Create per-detection record shared across publishing stages.
        class_name = class_name
        
        detection_entry = {
            "class_id": int(class_id),
            "instance_id": int(instance_id),
            "object_name": class_name,
            "confidence": float(confidence),
            "embedding": None,    # Placeholder
            "crop": None,         # Placeholder
            "mask_ros": eroded_mask
        }

        # Prepare CLIP crop (if current frame is scheduled for CLIP).
        if do_clip_frame:
            sx1, sy1, sx2, sy2 = CLIPProcessor.compute_square_crop(
                x1, y1, x2, y2, width, height, self.square_crop_scale
            )
            if sx2 > sx1 and sy2 > sy1:
                img_crop = cv_bgr[sy1:sy2, sx1:sx2]
                mask_crop = mask_uint8[sy1:sy2, sx1:sx2]

                if img_crop.size > 0 and mask_crop.size > 0:

                    kernel = np.ones((5, 5), np.uint8)
                    mask_crop = cv2.dilate(mask_crop, kernel, iterations=1)

                    # Gray masking
                    neutral_bg = np.full_like(img_crop, 122)  # neutral gray
                    
                    # The rest of your logic stays exactly the same
                    bg_mask = cv2.bitwise_not(mask_crop)
                    fg = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
                    bg = cv2.bitwise_and(neutral_bg, neutral_bg, mask=bg_mask)
                    final_crop = cv2.add(fg, bg)
                    detection_entry["crop"] = final_crop

        return detection_entry
    
    # ----------- Secondary Methods ------------ #

    def _load_clip_prompt(self):
        """
        Load clip_prompt from robot_command.json and update text embedding if changed.
        """
        try:
            # If command file does not exist yet, keep previous prompt.
            if not os.path.exists(self.robot_command_file):
                return
            
            with open(self.robot_command_file, 'r') as f:
                data = json.load(f)
            
            # Update goal prompt and regenerate text embedding only when prompt changes.
            clip_prompt = data.get('clip_prompts', None)
            if clip_prompt != self.current_clip_prompt:
                self.current_clip_prompt = clip_prompt
                self.goal_text_embedding = self.clip.encode_text(clip_prompt)
                self.get_logger().info(f"Updated Goal Ensemble: {clip_prompt}")

        except Exception as e:
            self.get_logger().error(f"Error loading robot_command.json: {e}")

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
        array_msg.header = header

        for det in detections:
            msg = DetectedObjectV3()
            
            # Basic Info
            msg.class_name = det["object_name"]
            msg.instance_id = int(det["instance_id"]) 
            msg.confidence = float(det.get("confidence", 0.0))
            
            # Embeddings
            current_emb = det["embedding"]
            msg.embedding = current_emb.tolist() if current_emb is not None else []
            msg.text_embedding = self.goal_text_embedding.tolist() if self.goal_text_embedding is not None else []

            # Compute goal similarity
            prob_goal = 0.0
            if current_emb is not None and self.goal_text_embedding is not None:
                prob_goal = self.clip.compute_sigmoid_probs(current_emb, self.goal_text_embedding)
                msg.similarity = float(prob_goal) if prob_goal is not None else 0.0
            else:
                msg.similarity = 0.0

            # ==========================================
            # SOTA EDGE FIX: Attach the 2D Mask
            # ==========================================
            if "mask_ros" in det and det["mask_ros"] is not None:
                # Convert the uint8 NumPy array to a ROS Image message.
                # 'mono8' is the standard encoding for single-channel grayscale/binary masks.
                msg.mask = self.bridge.cv2_to_imgmsg(det["mask_ros"], encoding="mono8")
            else:
                self.get_logger().warning(f"Detection {det['instance_id']} is missing 'mask_ros'")

            # Append to array
            array_msg.detections.append(msg)

            self.get_logger().debug(
                f"Packaged ID {det['instance_id']} ({det['object_name']}): Conf={det['confidence']:.3f}"
            )

        # Publish the entire frame of detections simultaneously
        self.detections_pub.publish(array_msg)
        self.get_logger().info(f"Published {len(array_msg.detections)} detections in batch.")

# -------------------- MAIN -------------------- # 

def main(args=None):
    rclpy.init(args=args)
    node = NoPCVisionNode()
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
