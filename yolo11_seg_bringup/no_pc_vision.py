import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Vector3
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
import message_filters

import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
import threading
import json
from time import perf_counter

from ultralytics import YOLO

from yolo11_seg_interfaces.msg import DetectedObject
from .utils.clip_processor import CLIPProcessor


# -------------------- CLASS ------------------- #

class NoPCVisionNode(Node):
    """
    ROS2 Node that handles Vision computation using camera's pointcloud
    """

    # ------------- Initialization ------------- #

    def __init__(self):
        super().__init__('no_pc_vision_node')
        self.get_logger().info("NoPCVisionNode initialized\n")

        # ============= Parameters ============= #

        # Communication parameters
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('pointcloud_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('enable_visualization', False)

        self.image_topic = self.get_parameter('image_topic').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.enable_vis = bool(self.get_parameter('enable_visualization').value)

        # YOLO parameters
        self.declare_parameter('model_path', '/home/workspace/yolo26m-seg.pt')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf', 0.55)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('retina_masks', True)

        self.model_path = self.get_parameter('model_path').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        self.retina_masks = bool(self.get_parameter('retina_masks').value)

        # CLIP parameterss
        self.declare_parameter('CLIP_model_name', 'ViT-B-16-SigLIP')
        self.declare_parameter('robot_command_file', '/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json')
        self.declare_parameter('map_file_path', '/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/map.json')
        self.declare_parameter('square_crop_scale', 1.2)

        self.CLIP_model_name = self.get_parameter('CLIP_model_name').value
        self.robot_command_file = self.get_parameter('robot_command_file').value
        self.map_file_path = self.get_parameter('map_file_path').value
        self.square_crop_scale = float(self.get_parameter('square_crop_scale').value)

        # =========== Initialization =========== #

        self.anno_topic = '/vision/annotated_image'
        self.markers_topic = '/vision/centroid_markers'
        self.detection_topic = '/vision/detections'

        self.frame_skip = 5
        self.prompt_check_interval = 760.0  # Check for new prompts every 760 seconds

        # Load YOLO model
        self.get_logger().info(f"Loading YOLO: {self.model_path}")
        self.model = YOLO(self.model_path, task='segment')    

        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on device: {self.device}\n")
        self.clip = CLIPProcessor(
            device=self.device, 
            model_name=self.CLIP_model_name, 
            pretrained="webli"
        )

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.current_pointcloud = None

        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_profile=qos_sensor)
        self.rgb_sub = message_filters.Subscriber(self, Image, self.image_topic, qos_profile=qos_sensor)
        self.pc_sub = message_filters.Subscriber(self, PointCloud2, self.pointcloud_topic, qos_profile=qos_sensor)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.pc_sub], 
            queue_size=50, 
            slop=0.5
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.get_logger().info(f"Subscribed to: {self.image_topic} and {self.pointcloud_topic}")

        # Publishers    
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.detections_pub = self.create_publisher(DetectedObject, self.detection_topic, 10)
        if self.enable_vis:
            self.vis_pub = self.create_publisher(Image, self.anno_topic, 10)

        # State Variables
        self.frame_count = 0
        self.class_colors = {}
        self.current_clip_prompt = None
        self.goal_text_embedding = None
        self.sync_lock = threading.Lock()
        
        # Timing instrumentation
        self.timing_stats = {
            'yolo_inference': [],
            'pointcloud_conversion': [],
            'detections_processing': [],
            'clip_encoding': [],
            'publishing': [],
            'total_frame': []
        }
        self.timing_window = 30  # Print stats every N frames

        self.CLASS_NAMES = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]

        self._load_clip_prompt()
        self.command_timer = self.create_timer(self.prompt_check_interval, self._load_clip_prompt)

        self.get_logger().info("Vision Node Ready.")

    # ---------------- Callbacks --------------- #

    def camera_info_cb(self, msg: CameraInfo):
        """ 
        Process camera intrinsic parameters 
        """
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(f"Camera intrinsics received: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def sync_callback(self, rgb_msg: Image, pc_msg: PointCloud2):
        """
        Synchronized RGB-PointCloud callback.
        """
        self.get_logger().info("Synchronized callback triggered", throttle_duration_sec=5.0)
        with self.sync_lock:
            self.current_pointcloud = pc_msg
            self.process_frame(rgb_msg, pc_msg)
   
    # --------------- Main Methods ------------- #

    def process_frame(self, rgb_msg: Image, pc_msg: PointCloud2):
        """
        Process a single RGB frame with pointcloud.
        """
        if self.fx is None:
            self.get_logger().warn("Waiting for camera intrinsics...", throttle_duration_sec=5.0)
            return
        
        self.frame_count += 1
        frame_start = perf_counter()

        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            height, width = cv_bgr.shape[:2]

            # ===== YOLO INFERENCE TIMING =====
            yolo_start = perf_counter()
            results = self.model.track(
                source=cv_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                retina_masks=self.retina_masks,
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
            
            self._process_detections(res, cv_bgr, pc_msg, rgb_msg, height, width)

            if self.enable_vis:
                # result.plot() draws boxes, masks, and IDs on the image
                annotated_frame = res.plot()
                vis_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                vis_msg.header = rgb_msg.header
                self.vis_pub.publish(vis_msg)
                self.get_logger().info(f"Published annotated image (frame {self.frame_count})", throttle_duration_sec=2.0)
            
            # ===== TOTAL FRAME TIMING =====
            total_time = (perf_counter() - frame_start) * 1000
            self.timing_stats['total_frame'].append(total_time)
            
            # Print timing stats every N frames
            if self.frame_count % self.timing_window == 0:
                self._print_timing_stats()
        
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def _process_detections(self, res, cv_bgr, pc_msg, rgb_msg, height, width):
        """ 
        Unified pipeline using camera's pointcloud
        """
        # ===== GPU-to-CPU TRANSFER TIMING =====
        gpu_transfer_start = perf_counter()
        xyxy = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.zeros(len(clss), dtype=float)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.arange(len(clss))
        
        # Transfer ALL masks at once (more efficient than per-detection)
        if hasattr(res, 'masks') and res.masks is not None:
            masks_np = res.masks.data.detach().cpu().numpy()
        else:
            masks_np = None
        
        gpu_transfer_time = (perf_counter() - gpu_transfer_start) * 1000
        self.get_logger().debug(f"GPU-to-CPU transfer: {gpu_transfer_time:.2f}ms")

        # ===== POINTCLOUD CONVERSION TIMING =====
        pc_start = perf_counter()
        try:
            # Create a structured NumPy data type to define the memory layout
            # This tells NumPy to look for x, y, z (floats) at specific offsets
            # 'itemsize' ensures we skip any extra fields (like intensity/color) correctly
            dtype_struct = np.dtype({
                'names': ['x', 'y', 'z'],
                'formats': ['<f4', '<f4', '<f4'],  # <f4 means Little Endian Float32
                'offsets': [0, 4, 8],              # Standard offsets for x, y, z
                'itemsize': pc_msg.point_step      # Step size helps jump over other data
            })

            # Read the raw bytes directly into a structured array (Zero-Copy usually)
            pc_array_structured = np.frombuffer(pc_msg.data, dtype=dtype_struct)

            # Stack them into a standard (N, 3) float array
            # This is the only "heavy" copy operation, but it's vectorized and fast
            pc_array = np.stack([
                pc_array_structured['x'], 
                pc_array_structured['y'], 
                pc_array_structured['z']
            ], axis=-1)

            # Reshape to image dimensions
            pc_array = pc_array.reshape(height, width, 3)

            pc_total_time = (perf_counter() - pc_start) * 1000
            
            self.timing_stats['pointcloud_conversion'].append(pc_total_time)
            self.get_logger().debug(f"PC Conversion Total: {pc_total_time:.2f}ms")

        except Exception as e:
            self.get_logger().error(f"Failed to convert pointcloud: {e}")
            return

        frame_detections = []
        batch_queue = []

        do_clip_frame = (self.frame_count % max(1, self.frame_skip) == 0)

        # ===== DETECTION PROCESSING TIMING =====
        det_process_start = perf_counter()
        for i in range(len(xyxy)):
            result = self.process_single_detection(
                i, xyxy[i], clss[i], ids[i],confs[i], masks_np,
                cv_bgr, pc_array,
                height, width, rgb_msg, do_clip_frame
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
        if batch_queue:
            try:
                clip_start = perf_counter()
                # Extract just the images for the batch
                images = [item["crop"] for item in batch_queue]
                
                # Run inference on all at once
                embeddings = self.clip.encode_images_batch(images)
                
                # Re-associate embeddings with metadata
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
        pub_start = perf_counter()
        self.publish_custom_detections(frame_detections, pc_msg.header, rgb_msg.header.stamp)
        self.publish_centroid_markers(frame_detections, pc_msg.header)
        if self.enable_vis:
            self.publish_centroid_markers(frame_detections, pc_msg.header)
        
        pub_time = (perf_counter() - pub_start) * 1000
        self.timing_stats['publishing'].append(pub_time)
        self.get_logger().debug(f"Publishing: {pub_time:.2f}ms")

    def process_single_detection(self, idx, bbox, class_id, instance_id, confidence, masks_np,
                                 cv_bgr, pc_array,
                                 height, width, rgb_msg, do_clip_frame):
        """ 
        Process geometry using camera's pointcloud and prepare data container. 
        Returns detection_dictionary
        """
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width - 1))
        y2 = max(0, min(int(y2), height - 1))

        if x2 <= x1 or y2 <= y1:
            return None
        
        if masks_np is None or idx >= masks_np.shape[0]:
            return None

        # Get mask for this instance (already on CPU)
        m = masks_np[idx]
        if m.shape[0] != height or m.shape[1] != width:
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        binary_mask = (m > 0.5)

        # Extract points from the pointcloud using the mask
        masked_points = pc_array[binary_mask]
        
        # Filter out invalid points (NaN or zero)
        valid_mask = ~np.isnan(masked_points).any(axis=1)
        valid_points = masked_points[valid_mask]
        
        if len(valid_points) == 0:
            return None
        
        # Compute centroid
        centroid = np.mean(valid_points, axis=0)
        centroid = (float(centroid[0]), float(centroid[1]), float(centroid[2]))

        min_box = np.min(valid_points, axis=0) # [min_x, min_y, min_z]
        max_box = np.max(valid_points, axis=0) # [max_x, max_y, max_z]

        # --- Create Shared Object ---
        class_name = self.class_id_to_name(int(class_id))

        detection_entry = {
            "class_id": int(class_id),
            "instance_id": int(instance_id),
            "object_name": class_name,
            "confidence": float(confidence),
            "centroid": centroid, # (x, y, z) tuple
            "embedding": None,    # Placeholder
            "crop": None,         # Placeholder
            "box_min": (float(min_box[0]), float(min_box[1]), float(min_box[2])),
            "box_max": (float(max_box[0]), float(max_box[1]), float(max_box[2])),
        }

        # --- Prepare CLIP Crop (If enabled) ---
        if do_clip_frame:
            sx1, sy1, sx2, sy2 = CLIPProcessor.compute_square_crop(
                x1, y1, x2, y2, width, height, self.square_crop_scale
            )
            if sx2 > sx1 and sy2 > sy1:
                binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)

                img_crop = cv_bgr[sy1:sy2, sx1:sx2]
                mask_crop = binary_mask_uint8[sy1:sy2, sx1:sx2]

                if img_crop.size > 0 and mask_crop.size > 0:
                    # Gray masking
                    neutral_bg = np.full_like(img_crop, 122)  # neutral gray
                    bg_mask = cv2.bitwise_not(mask_crop)
                    fg = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
                    bg = cv2.bitwise_and(neutral_bg, neutral_bg, mask=bg_mask)
                    final_crop = cv2.add(fg, bg)
                    detection_entry["crop"] = final_crop

        return detection_entry
    
    # ----------- Secondary Methods ------------ #

    @staticmethod
    def get_color_for_class(class_id: str, class_colors: dict):
        """ 
        Deterministically generate a color for a given class ID. 
        """
        if class_id not in class_colors:
            h = abs(hash(class_id))
            r = (h >> 0) & 0xFF
            g = (h >> 8) & 0xFF
            b = (h >> 16) & 0xFF
            if r < 30 and g < 30 and b < 30:
                r = (r + 128) & 0xFF
                g = (g + 64) & 0xFF
            class_colors[class_id] = (r, g, b)
        return class_colors[class_id]

    def _load_clip_prompt(self):
        """
        Load clip_prompt from robot_command.json and update text embedding if changed.
        """
        try:
            if not os.path.exists(self.robot_command_file):
                return
            
            with open(self.robot_command_file, 'r') as f:
                data = json.load(f)
            
            # Update GOAL Prompt
            clip_prompt = data.get('clip_prompt', None)
            if clip_prompt != self.current_clip_prompt:
                self.current_clip_prompt = clip_prompt
                self.goal_text_embedding = self.clip.encode_text(clip_prompt)
                self.get_logger().info(f"Updated Goal Ensemble: {clip_prompt}")

        except Exception as e:
            self.get_logger().error(f"Error loading robot_command.json: {e}")

    def class_id_to_name(self, class_id: int) -> str:
        """
        Convert class ID to class name.
        """
        if 0 <= class_id < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id]
        return f"class_{class_id}"
    
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

    def publish_custom_detections(self, detections, header, timestamp):
        """ Iterate through the unified list and publish messages. """
        for det in detections:
            msg = DetectedObject()
            
            # Basic Info
            msg.object_name = det["object_name"]
            msg.object_id = det["instance_id"]
            msg.confidence = float(det.get("confidence", 0.0))
            
            # Centroid
            cx, cy, cz = det["centroid"]
            msg.centroid = Vector3(x=cx, y=cy, z=cz)
            
            # Timestamp (Extract directly from the header)
            msg.timestamp = header.stamp
            
            current_emb = det["embedding"]
            msg.embedding = current_emb.tolist() if current_emb is not None else []

            # Only compute similarity if embedding is available
            prob_goal = 0.0
            if current_emb is not None and self.goal_text_embedding is not None:
                prob_goal = self.clip.compute_sigmoid_probs(current_emb, self.goal_text_embedding)
                msg.similarity = float(prob_goal) if prob_goal is not None else 0.0
            else:
                msg.similarity = 0.0

            self.get_logger().info(
                f"ID {det['instance_id']} ({det['object_name']}): Goal={prob_goal}"
            )

            # --- NEW: Assign Box ---
            msg.box_min.x = det["box_min"][0]
            msg.box_min.y = det["box_min"][1]
            msg.box_min.z = det["box_min"][2]
            
            msg.box_max.x = det["box_max"][0]
            msg.box_max.y = det["box_max"][1]
            msg.box_max.z = det["box_max"][2]

            # Add text embedding for mapper convenience
            msg.text_embedding = self.goal_text_embedding.tolist() if self.goal_text_embedding is not None else []
            self.detections_pub.publish(msg)

    def publish_centroid_markers(self, detections, header):
        """Publish centroids as marker array."""
        try:
            marker_array = MarkerArray()
            
            for i, det in enumerate(detections):
                marker = Marker()
                marker.header = header
                marker.ns = "centroids"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # Set position from centroid
                cx, cy, cz = det["centroid"]
                marker.pose.position.x = cx
                marker.pose.position.y = cy
                marker.pose.position.z = cz
                marker.pose.orientation.w = 1.0
                
                # Set scale (radius of sphere)
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                
                # Set color from class color
                class_id_str = str(det["class_id"])
                r, g, b = self.get_color_for_class(class_id_str, self.class_colors)
                marker.color.r = r / 255.0
                marker.color.g = g / 255.0
                marker.color.b = b / 255.0
                marker.color.a = 1.0
                
                marker_array.markers.append(marker)
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing centroid markers: {e}")

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
