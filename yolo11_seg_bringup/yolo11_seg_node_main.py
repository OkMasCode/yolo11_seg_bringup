#!/usr/bin/env python3
"""
Refactored YOLO segmentation node with modular structure.
Main node class that orchestrates YOLO inference, CLIP embeddings, and pointcloud generation.
"""
import numpy as np
import threading
import json
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
from yolo11_seg_interfaces.msg import DetectedObject
from ultralytics import YOLO
import torch

from .utils.pointcloud import PointCloudProcessor
from .utils.clip_processor import CLIPProcessor
from .utils.visualization import Visualizer

# Define COCO class names
CLASS_NAMES = [
    "person",         # 0
    "bicycle",        # 1
    "car",            # 2
    "motorcycle",     # 3
    "airplane",       # 4
    "bus",            # 5
    "train",          # 6
    "truck",          # 7
    "boat",           # 8
    "traffic light",  # 9
    "fire hydrant",   # 10
    "stop sign",      # 11
    "parking meter",  # 12
    "bench",          # 13
    "bird",           # 14
    "cat",            # 15
    "dog",            # 16
    "horse",          # 17
    "sheep",          # 18
    "cow",            # 19
    "elephant",       # 20
    "bear",           # 21
    "zebra",          # 22
    "giraffe",        # 23
    "backpack",       # 24
    "umbrella",       # 25
    "handbag",        # 26
    "tie",            # 27
    "suitcase",       # 28
    "frisbee",        # 29
    "skis",           # 30
    "snowboard",      # 31
    "sports ball",    # 32
    "kite",           # 33
    "baseball bat",   # 34
    "baseball glove", # 35
    "skateboard",     # 36
    "surfboard",      # 37
    "tennis racket",  # 38
    "bottle",         # 39
    "wine glass",     # 40
    "cup",            # 41
    "fork",           # 42
    "knife",          # 43
    "spoon",          # 44
    "bowl",           # 45
    "banana",         # 46
    "apple",          # 47
    "sandwich",       # 48
    "orange",         # 49
    "broccoli",       # 50
    "carrot",         # 51
    "hot dog",        # 52
    "pizza",          # 53
    "donut",          # 54
    "cake",           # 55
    "chair",          # 56
    "couch",          # 57
    "potted plant",   # 58
    "bed",            # 59
    "dining table",   # 60
    "toilet",         # 61
    "tv",             # 62
    "laptop",         # 63
    "mouse",          # 64
    "remote",         # 65
    "keyboard",       # 66
    "cell phone",     # 67
    "microwave",      # 68
    "oven",           # 69
    "toaster",        # 70
    "sink",           # 71
    "refrigerator",   # 72
    "book",           # 73
    "clock",          # 74
    "vase",           # 75
    "scissors",       # 76
    "teddy bear",     # 77
    "hair drier",     # 78
    "toothbrush",     # 79
]


class Yolo11SegNode(Node):
    """ROS2 node for YOLO segmentation with CLIP embeddings and pointcloud generation."""
    
    def __init__(self):
        super().__init__("yolo11_seg_node")
        
        # Declare and get parameters
        self._declare_parameters()
        self._get_parameters()
        
        # Initialize models and processors
        self._initialize_models()
        self._initialize_processors()
        
        # Setup ROS communications
        self._setup_subscriptions()
        self._setup_publishers()
        
        # State variables
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth_msg = None
        self.sync_lock = threading.Lock()
        self.class_colors = {}
        
        # Detection results storage
        self.last_clip_embeddings = []
        self.last_centroids = []
        self.last_detection_meta = []
        
        self.get_logger().info(f"Ready. Publishing to {self.pc_topic}")
    
    def _declare_parameters(self):
        """Declare all ROS parameters."""
        # Topic parameters
        self.declare_parameter("model_path", "/home/sensor/yolov8n-seg.engine")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("pointcloud_topic", "/yolo/pointcloud")
        self.declare_parameter("annotated_topic", "/yolo/annotated")
        self.declare_parameter("clip_boxes_topic", "/yolo/clip_boxes")
        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("text_prompt", "a photo of a person")
        
        # Algorithm parameters
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.70)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("retina_masks", True) # use retina masks for higher resolution
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("pc_downsample", 2) # factor for downsampling pointcloud
        self.declare_parameter("pc_max_range", 8.0) # meters
        self.declare_parameter("mask_threshold", 0.5)
        self.declare_parameter("clip_square_scale", 1.4) # scale factor for enlarging clip box around detection
        self.declare_parameter("publish_annotated", False)
        self.declare_parameter("publish_clip_boxes_vis", False)
    
    def _get_parameters(self):
        """Get all parameter values."""
        # Topics
        self.model_path = self.get_parameter("model_path").value
        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.pc_topic = self.get_parameter("pointcloud_topic").value
        self.anno_topic = self.get_parameter("annotated_topic").value
        self.clip_boxes_topic = self.get_parameter("clip_boxes_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.text_prompt = self.get_parameter("text_prompt").value
        
        # Algorithm settings
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.retina_masks = bool(self.get_parameter("retina_masks").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.pc_downsample = int(self.get_parameter("pc_downsample").value)
        self.pc_max_range = float(self.get_parameter("pc_max_range").value)
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        self.clip_square_scale = float(self.get_parameter("clip_square_scale").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.publish_clip_boxes_vis = bool(self.get_parameter("publish_clip_boxes_vis").value)
    
    def _load_clip_prompt_from_command(self):
        """
        Attempt to load the clip_prompt from robot_command.json.
        Falls back to the text_prompt parameter if file doesn't exist or is invalid.
        
        Returns:
            str: The clip_prompt from robot_command.json, or self.text_prompt as fallback
        """
        command_file = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "config", "robot_command.json"
        )
        
        try:
            if os.path.exists(command_file):
                with open(command_file, 'r') as f:
                    command_data = json.load(f)
                    if 'clip_prompt' in command_data and command_data['clip_prompt']:
                        prompt = command_data['clip_prompt']
                        self.get_logger().info(f"Loaded clip_prompt from robot_command.json: '{prompt}'")
                        return prompt
            else:
                self.get_logger().warn(f"robot_command.json not found at {command_file}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load robot_command.json: {e}")
        
        # Fallback to parameter
        self.get_logger().info(f"Using text_prompt parameter: '{self.text_prompt}'")
        return self.text_prompt
    
    def _initialize_models(self):
        """Initialize YOLO and CLIP models."""
        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path, task="segment")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on {self.device}...")
        self.clip_processor = CLIPProcessor(device=self.device)
        
        # Load clip_prompt from robot_command.json if available, otherwise use text_prompt parameter
        self.text_prompt = self._load_clip_prompt_from_command()
        
        # Encode text prompt
        self.text_features = self.clip_processor.encode_text_prompt(self.text_prompt)
        self.text_embedding_list = self.text_features.cpu().numpy().flatten().tolist()
        self.get_logger().info(f"Searching for: '{self.text_prompt}'")
    
    def _initialize_processors(self):
        """Initialize processing utilities."""
        self.bridge = CvBridge()
        self.visualizer = Visualizer()
        self.pc_processor = None  # Will be initialized after camera info received
    
    @staticmethod
    def class_id_to_name(class_id: int) -> str:
        """
        Convert a class ID to its corresponding class name.
        If the class ID is out of range, return a generic name.
        """
        if 0 <= class_id < len(CLASS_NAMES):
            return CLASS_NAMES[class_id]
        return f"class_{class_id}"
    
    def _setup_subscriptions(self):
        """Setup ROS subscriptions."""
        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.rgb_sub = self.create_subscription(
            Image, self.image_topic, self.rgb_callback, qos_profile=qos_sensor
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile=qos_sensor
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_profile=qos_sensor
        )
    
    def _setup_publishers(self):
        """Setup ROS publishers."""
        self.marker_pub = self.create_publisher(MarkerArray, "/yolo/centroids", 10)
        self.pc_pub = self.create_publisher(PointCloud2, self.pc_topic, 10)
        self.anno_pub = self.create_publisher(Image, self.anno_topic, 10)
        self.clip_boxes_pub = self.create_publisher(Image, self.clip_boxes_topic, 10)
        self.detections_pub = self.create_publisher(DetectedObject, self.detections_topic, 10)
    
    def camera_info_cb(self, msg: CameraInfo):
        """Process camera intrinsic parameters."""
        if self.fx is None:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info(
                f"Camera intrinsics set: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
            
            # Initialize pointcloud processor now that we have intrinsics
            self.pc_processor = PointCloudProcessor(
                self.fx, self.fy, self.cx, self.cy,
                device=self.device,
                depth_scale=self.depth_scale,
                pc_downsample=self.pc_downsample,
                pc_max_range=self.pc_max_range,
                mask_threshold=self.mask_threshold
            )
    
    def depth_callback(self, msg: Image):
        """Store latest depth message."""
        with self.sync_lock:
            self.latest_depth_msg = msg
    
    def rgb_callback(self, msg: Image):
        """Process RGB image with synchronized depth."""
        with self.sync_lock:
            if self.latest_depth_msg is None:
                return
            rgb_msg = msg
            depth_msg = self.latest_depth_msg
        
        self.process_frame(rgb_msg, depth_msg)
    
    def process_frame(self, rgb_msg, depth_msg):
        """
        Main processing pipeline for RGB-D frame.
        
        Args:
            rgb_msg: RGB image message
            depth_msg: Depth image message
        """
        try:
            cb_start = self.get_clock().now()
            
            # Convert images
            frame_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            
            if depth_img.ndim != 2 or self.fx is None:
                return
            
            height, width = depth_img.shape
            
            # YOLO inference
            yolo_start = self.get_clock().now()
            results = self.model.track(
                source=frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                retina_masks=self.retina_masks,
                stream=False,
                verbose=False,
                persist=True,
                tracker="botsort.yaml",
            )
            yolo_time = (self.get_clock().now() - yolo_start).nanoseconds / 1e9
            self.get_logger().info(f"YOLO inference time: {yolo_time:.3f} seconds, {1.0/yolo_time:.2f} FPS")
            
            res = results[0]
            if not hasattr(res, "boxes") or len(res.boxes) == 0:
                return
            
            # Process detections
            pc_start = self.get_clock().now()
            self._process_detections(res, frame_bgr, depth_img, depth_msg, rgb_msg, height, width)
            
            pc_time = (self.get_clock().now() - pc_start).nanoseconds / 1e9
            self.get_logger().info(f"Pointcloud computation time: {pc_time:.3f} seconds, {1.0/pc_time:.2f} FPS")
            
            # Publish visualization
            self._publish_visualizations(res, rgb_msg)
            
            # Publish results
            self._publish_results(rgb_msg.header.stamp, depth_msg.header.frame_id, depth_msg.header)
            
            total_time = (self.get_clock().now() - cb_start).nanoseconds / 1e9
            self.get_logger().info(f"End-to-end processing time: {total_time:.3f} seconds, {1.0/total_time:.2f} FPS")
            
        except Exception as e:
            self.get_logger().error(f"Error in process_frame: {e}")
    
    def _process_detections(self, res, frame_bgr, depth_img, depth_msg, rgb_msg, height, width):
        """Process all detections from YOLO results."""
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        clss = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.zeros(len(clss))
        masks_t = res.masks.data if hasattr(res, "masks") and res.masks is not None else None
        
        is_uint16 = (depth_msg.encoding == "16UC1")
        scale_factor = (1.0 / self.depth_scale) if is_uint16 else 1.0
        
        device = masks_t.device if masks_t is not None else self.device
        depth_t, valid_mask_t = self.pc_processor.prepare_depth_tensor(depth_img, depth_msg.encoding, scale_factor)
        
        # Reset frame data
        self.last_clip_embeddings = []
        self.last_centroids = []
        self.last_detection_meta = []
        
        all_points_list = []
        clip_boxes_vis = None
        
        for i in range(len(xyxy)):
            result = self._process_single_detection(
                i, xyxy[i], clss[i], ids[i], masks_t,
                frame_bgr, depth_t, valid_mask_t,
                scale_factor, height, width, rgb_msg
            )
            
            if result is None:
                continue
            
            instance_cloud_t, clip_crop_info = result
            all_points_list.append(instance_cloud_t)
            
            # Handle CLIP visualization (only draw if publishing is enabled)
            if clip_crop_info is not None and self.publish_clip_boxes_vis:
                if clip_boxes_vis is None:
                    clip_boxes_vis = frame_bgr.copy()
                self.visualizer.draw_clip_boxes(
                    clip_boxes_vis,
                    clip_crop_info["sx1"], clip_crop_info["sy1"],
                    clip_crop_info["sx2"], clip_crop_info["sy2"],
                    clip_crop_info["x1"], clip_crop_info["y1"],
                    clip_crop_info["x2"], clip_crop_info["y2"],
                    clip_crop_info["instance_id"],
                    clip_crop_info["class_id"]
                )
        
        # Publish pointcloud
        if all_points_list:
            final_points_t = torch.cat(all_points_list, dim=0)
            final_points = final_points_t.detach().cpu().numpy().astype(np.float32)
            PointCloudProcessor.publish_pointcloud(final_points, depth_msg.header, self.pc_pub)
        
        # Store visualization
        self._clip_boxes_vis = clip_boxes_vis
    
    def _process_single_detection(self, idx, bbox, class_id, instance_id, masks_t,
                                  frame_bgr, depth_t, valid_mask_t, scale_factor,
                                  height, width, rgb_msg):
        """Process a single detection."""
        x1, y1, x2, y2 = bbox
        
        # Clamp to image bounds
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Compute CLIP crop
        sx1, sy1, sx2, sy2 = CLIPProcessor.compute_square_crop(
            x1, y1, x2, y2, width, height, self.clip_square_scale
        )
        
        clip_crop_info = None
        if sx2 > sx1 and sy2 > sy1:
            # Process CLIP embedding
            crop_bgr = frame_bgr[sy1:sy2, sx1:sx2]
            try:
                embedding = self.clip_processor.encode_image_crop(crop_bgr)
                self.last_clip_embeddings.append({
                    "class_id": int(class_id),
                    "instance_id": int(instance_id),
                    "bbox_full": [int(x1), int(y1), int(x2), int(y2)],
                    "square_crop": [int(sx1), int(sy1), int(sx2), int(sy2)],
                    "embedding": embedding,
                })
                
                clip_crop_info = {
                    "sx1": sx1, "sy1": sy1, "sx2": sx2, "sy2": sy2,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "class_id": int(class_id),
                    "instance_id": int(instance_id)
                }
            except Exception as e:
                self.get_logger().warn(f"CLIP embedding failed for det {idx}: {e}")
        
        # Process pointcloud
        if masks_t is None or idx >= masks_t.shape[0]:
            return None
        
        rgb_color = self.visualizer.get_color_for_class(str(class_id), self.class_colors)
        instance_cloud_t, centroid = self.pc_processor.process_detection(
            masks_t[idx], depth_t, valid_mask_t,
            int(class_id), int(instance_id),
            rgb_color, scale_factor
        )
        
        if instance_cloud_t is None:
            return None
        
        # Store centroid and metadata
        self.last_centroids.append({
            "class_id": int(class_id),
            "instance_id": int(instance_id),
            "centroid": centroid
        })
        
        class_name = self.class_id_to_name(int(class_id))
        timestamp = Time(sec=rgb_msg.header.stamp.sec, nanosec=rgb_msg.header.stamp.nanosec)
        self.last_detection_meta.append({
            "name": class_name,
            "instance_id": int(instance_id),
            "timestamp": timestamp,
        })
        
        return instance_cloud_t, clip_crop_info
    
    def _publish_visualizations(self, res, rgb_msg):
        """Publish visualization images."""
        # Annotated image (only if parameter is enabled)
        if self.publish_annotated:
            try:
                annotated_bgr = res.plot()
                anno_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding="bgr8")
                anno_msg.header = rgb_msg.header
                self.anno_pub.publish(anno_msg)
            except Exception as e:
                self.get_logger().warn(f"Annotated publish failed: {e}")
        
        # CLIP boxes visualization (only if parameter is enabled)
        if self.publish_clip_boxes_vis and hasattr(self, '_clip_boxes_vis') and self._clip_boxes_vis is not None:
            try:
                clip_msg = self.bridge.cv2_to_imgmsg(self._clip_boxes_vis, encoding="bgr8")
                clip_msg.header = rgb_msg.header
                self.clip_boxes_pub.publish(clip_msg)
            except Exception as e:
                self.get_logger().warn(f"CLIP boxes publish failed: {e}")
    
    def _publish_results(self, stamp, frame_id, header):
        """Publish centroids and detections."""
        # Publish centroids
        try:
            if self.last_centroids:
                marker_array = MarkerArray()
                for idx, entry in enumerate(self.last_centroids):
                    cx, cy, cz = entry["centroid"]
                    class_id = entry["class_id"]
                    class_name = self.class_id_to_name(class_id)
                    centroid_vec = Vector3(x=float(cx), y=float(cy), z=float(cz))
                    
                    color_rgb = self.visualizer.get_color_for_class(str(class_id), self.class_colors)
                    marker = self.visualizer.create_centroid_marker(
                        class_name, centroid_vec, class_id, idx, frame_id, stamp, color_rgb
                    )
                    marker_array.markers.append(marker)
                
                self.marker_pub.publish(marker_array)
        except Exception as e:
            self.get_logger().warn(f"Centroid markers publish failed: {e}")
        
        # Publish detections
        try:
            if not self.last_detection_meta:
                return
            
            for meta in self.last_detection_meta:
                instance_id = meta["instance_id"]
                
                # Find centroid
                centroid_vec = None
                for centroid_entry in self.last_centroids:
                    if centroid_entry["instance_id"] == instance_id:
                        cx, cy, cz = centroid_entry["centroid"]
                        centroid_vec = Vector3(x=float(cx), y=float(cy), z=float(cz))
                        break
                
                if centroid_vec is None:
                    continue
                
                # Find CLIP embedding
                embedding_array = []
                for clip_entry in self.last_clip_embeddings:
                    if clip_entry["instance_id"] == instance_id:
                        embedding_array = clip_entry["embedding"].tolist()
                        break
                
                # Publish detection message
                msg = DetectedObject()
                msg.object_name = meta["name"]
                msg.object_id = instance_id
                msg.centroid = centroid_vec
                msg.timestamp = meta["timestamp"]
                msg.embedding = embedding_array
                msg.text_embedding = self.text_embedding_list
                
                self.detections_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Detections publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = Yolo11SegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
