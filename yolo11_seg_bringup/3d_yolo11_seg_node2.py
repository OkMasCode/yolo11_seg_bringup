#!/usr/bin/env python3
import struct
import numpy as np
import threading
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from yolo11_seg_interfaces.msg import DetectedObject
from ultralytics import YOLO
import torch
import clip
import cv2
from PIL import Image as PILImage

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
    def __init__(self):
        super().__init__("yolo11_seg_node")

        # ============= Parameters ============= #
        self.declare_parameter("model_path", "/home/sensor/yolov8n-seg.engine")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("pointcloud_topic", "/yolo/pointcloud")
        self.declare_parameter("annotated_topic", "/yolo/annotated")
        self.declare_parameter("clip_boxes_topic", "/yolo/clip_boxes")
        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("text_prompt", "a photo of a person}")

        self.declare_parameter("similarity_threshold", 0.3)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.70)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("retina_masks", True)
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("pc_downsample", 2)
        self.declare_parameter("pc_max_range", 8.0)
        self.declare_parameter("mask_threshold", 0.5)
        self.declare_parameter("clip_square_scale", 1.4)
        self.declare_parameter("debug_clip_boxes", False)

        model_path = self.get_parameter("model_path").value
        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.pc_topic = self.get_parameter("pointcloud_topic").value
        self.anno_topic = self.get_parameter("annotated_topic").value
        self.clip_boxes_topic = self.get_parameter("clip_boxes_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.text_prompt = self.get_parameter("text_prompt").value

        self.similarity_threshold = float(self.get_parameter("similarity_threshold").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.retina_masks = bool(self.get_parameter("retina_masks").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.pc_downsample = int(self.get_parameter("pc_downsample").value)
        self.pc_max_range = float(self.get_parameter("pc_max_range").value)
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        self.clip_square_scale = float(self.get_parameter("clip_square_scale").value)
        self.debug_clip_boxes = bool(self.get_parameter("debug_clip_boxes").value)

        # ============= Initialization ============= #

        self.get_logger().info(f"Loading model: {model_path}")
        self.model = YOLO(model_path, task="segment")

        qos_sensor = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

        self.bridge = CvBridge()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on {self.device}...")
        self.model2, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.text_features = self.encode_text_prompt(self.text_prompt)
        self.text_embedding_list = self.text_features.cpu().numpy().flatten().tolist()
        self.get_logger().info(f"Searching for: '{self.text_prompt}'")

        self.last_clip_embeddings = []  # list of dicts for the latest frame (not published)
        self.last_centroids = []  # list of dicts: {'class_id','instance_id','centroid':(x,y,z)} for latest frame
        self.last_detection_meta = []  # list of dicts: {'name','instance_id','timestamp'} for latest frame
 
        self.latest_depth_msg = None
        self.sync_lock = threading.Lock() # To protect access to latest_depth_msg

        self.rgb_sub = self.create_subscription(Image, self.image_topic, self.rgb_callback, qos_profile=qos_sensor)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile=qos_sensor)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_profile=qos_sensor)

        self.marker_pub = self.create_publisher(MarkerArray, "/yolo/centroids", 10)
        self.pc_pub = self.create_publisher(PointCloud2, self.pc_topic, 10)
        self.anno_pub = self.create_publisher(Image, self.anno_topic, 10)
        self.clip_boxes_pub = self.create_publisher(Image, self.clip_boxes_topic, 10)
        self.detections_pub = self.create_publisher(DetectedObject, self.detections_topic, 10)

        self.fx = self.fy = self.cx = self.cy = None
        self.fx_t = self.fy_t = self.cx_t = self.cy_t = None  # Cached tensors
        self.class_colors = {}

        self.get_logger().info(f"Ready. Publishing to {self.pc_topic}")

    def encode_text_prompt(self, text):
        with torch.no_grad():
            text_token = clip.tokenize([text]).to(self.device)
            
            # FIX: Use self.model2 (CLIP), not self.model (YOLO)
            text_features = self.model2.encode_text(text_token)
            
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def camera_info_cb(self, msg: CameraInfo):
        """
        CameraInfo callback.

        Reads the intrinsic parameters (fx, fy, cx, cy) from the incoming
        CameraInfo message the first time it is received and caches them
        in the node.

        These intrinsics come from the 3x3 camera matrix K:
            [ fx  0  cx ]
            [  0 fy  cy ]
            [  0  0   1 ]
        """
        if self.fx is None:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info(
                f"Camera intrinsics set: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )

    def depth_callback(self, msg: Image):
        """
        Store the latest depth message with timestamp.
        """
        with self.sync_lock:
            self.latest_depth_msg = msg

    def rgb_callback(self, msg: Image):
        """
        On RGB arrival, process with latest depth if available and timestamp is close.
        """
        with self.sync_lock:
            if self.latest_depth_msg is None:
                return
            rgb_msg = msg
            depth_msg = self.latest_depth_msg
        self.synced_cb(rgb_msg, depth_msg)

    def get_color_for_class(self, class_id: str):
        """
        Deterministically map a class_id string to an RGB color.
        """
        if class_id not in self.class_colors:
            h = abs(hash(class_id))
            r = (h >> 0) & 0xFF
            g = (h >> 8) & 0xFF
            b = (h >> 16) & 0xFF
            if r < 30 and g < 30 and b < 30:
                r = (r + 128) & 0xFF
                g = (g + 64) & 0xFF
            self.class_colors[class_id] = (r, g, b)
        return self.class_colors[class_id]

    @staticmethod
    def class_id_to_name(class_id: int) -> str:
        """
        Convert a class ID to its corresponding class name.
        If the class ID is out of range, return a generic name.
        """
        if 0 <= class_id < len(CLASS_NAMES):
            return CLASS_NAMES[class_id]
        return f"class_{class_id}"

    @staticmethod
    def pack_rgb(r: int, g: int, b: int) -> float:
        """
        Pack 3x uint8 RGB into float32 for PointCloud2 'rgb' field.
        Args:
            r: Red channel (0-255)
            g: Green channel (0-255)
            b: Blue channel (0-255)
        Returns:
            float32 representing packed RGB value
        """
        rgb_uint32 = (r << 16) | (g << 8) | b
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]

    def synced_cb(self, rgb_msg, depth_msg):

        """
        - Perform YOLOv11-seg inference on RGB image
        - Filter out dynamic classes
        - remove points with invalid depth
        - Remove masks with too few points
        - Filter points by depth median
        - Generate colored 3D point cloud with class and instance IDs
        - Publish point cloud and annotated image
        """
        try:
            # Start end-to-end timing
            cb_start = self.get_clock().now()

            # Convert ROS image messages into NumPy/OpenCV images
            frame_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            if depth_img.ndim != 2 or self.fx is None:
                return

            height, width = depth_img.shape

            # YOLOv11-seg model inference on ROI only
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
            yolo_end = self.get_clock().now()
            yolo_time = (yolo_end - yolo_start).nanoseconds / 1e9
            self.get_logger().info(f"YOLO inference time: {yolo_time:.3f} seconds, {1.0/yolo_time:.2f} FPS")

            # Process the first result
            res = results[0]

            if not hasattr(res, "boxes") or len(res.boxes) == 0: # No detections
                return

            # Start pointcloud computation timing
            pc_start = self.get_clock().now()

            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids  = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.zeros(len(clss))
            masks_t = res.masks.data if hasattr(res, "masks") and res.masks is not None else None

            all_points_list = []

            step = max(self.pc_downsample, 1)
            DEPTH_TOLERANCE = 0.5
            MIN_POINTS = 10

            # DYNAMIC_CLASSES = {0, 1, 2, 3, 5, 7, 15, 16}
            
            is_uint16 = (depth_msg.encoding == "16UC1") # If depth is in uint16 format it is in mm
            scale_factor = (1.0 / self.depth_scale) if is_uint16 else 1.0 # Convert from mm to m

            # Choose device based on YOLO masks if available, else prefer CUDA
            device = (
                masks_t.device
                if masks_t is not None and isinstance(masks_t, torch.Tensor)
                else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            )

            # Cache intrinsic tensors on first use
            if self.fx_t is None or self.fx_t.device != device:
                self.fx_t = torch.tensor(self.fx, dtype=torch.float32, device=device)
                self.fy_t = torch.tensor(self.fy, dtype=torch.float32, device=device)
                self.cx_t = torch.tensor(self.cx, dtype=torch.float32, device=device)
                self.cy_t = torch.tensor(self.cy, dtype=torch.float32, device=device)

            depth_t = torch.from_numpy(depth_img.astype(np.float32)).to(device)
            valid_mask_t = (depth_t > 0) & (~torch.isnan(depth_t))
            if self.pc_max_range > 0.0:
                valid_mask_t = valid_mask_t & (depth_t * scale_factor <= self.pc_max_range)

            # Reset CLIP embeddings list for this frame
            self.last_clip_embeddings = []
            # Reset centroids list for this frame
            self.last_centroids = []
            # Reset detection meta list for this frame
            self.last_detection_meta = []
            
            # Create visualization image for CLIP boxes (lazy copy - only if needed)
            clip_boxes_vis = None

            for i in range(len(xyxy)): # Iterate over detections (ROI coords)
                
                class_id = int(clss[i])

                # Skip to next if dynamic class for point cloud generation
                # if class_id in DYNAMIC_CLASSES:
                #     continue

                x1, y1, x2, y2 = xyxy[i]
                
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                if x2 <= x1 or y2 <= y1:
                    continue # Invalid / empty box
                
                # Compute a square crop at least 30% bigger than bbox for CLIP
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                side = max(bw, bh)
                side = int(np.ceil(side * max(1.3, float(self.clip_square_scale))))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                half = side / 2.0
                sx1 = int(np.floor(cx - half))
                sy1 = int(np.floor(cy - half))
                sx2 = int(np.ceil(cx + half))
                sy2 = int(np.ceil(cy + half))
                sx1 = max(0, min(sx1, width - 1))
                sy1 = max(0, min(sy1, height - 1))
                sx2 = max(0, min(sx2, width - 1))
                sy2 = max(0, min(sy2, height - 1))

                # Store crop info for later CLIP processing (deferred to avoid blocking)
                clip_crop_info = None
                if sx2 > sx1 and sy2 > sy1:
                    clip_crop_info = {
                        "sx1": sx1, "sy1": sy1, "sx2": sx2, "sy2": sy2,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "class_id": class_id,
                        "instance_id": int(ids[i]) if isinstance(ids, np.ndarray) else 0
                    }

                if x2 <= x1 or y2 <= y1:
                    continue # Invalid / empty box

                # Object support mask: segmentation if available, else rectangular bbox
                if masks_t is None or i >= masks_t.shape[0]:
                    continue

                obj_mask_t = (masks_t[i] >= self.mask_threshold).to(device)
                valid_t = valid_mask_t & obj_mask_t

                v_coords_t, u_coords_t = valid_t.nonzero(as_tuple=True)
                if v_coords_t.numel() < MIN_POINTS:
                    continue

                z_vals_t = (depth_t[v_coords_t, u_coords_t] * scale_factor).to(torch.float32)

                # Fast outlier rejection using percentiles (faster than median)
                z_min = torch.quantile(z_vals_t, 0.1)
                z_max = torch.quantile(z_vals_t, 0.9)
                keep_mask_t = (z_vals_t >= z_min) & (z_vals_t <= z_max)
                if not torch.any(keep_mask_t):
                    continue

                z_clean_t = z_vals_t[keep_mask_t]
                u_clean_t = u_coords_t[keep_mask_t].to(torch.float32)
                v_clean_t = v_coords_t[keep_mask_t].to(torch.float32)

                # Optional downsampling for speed
                if self.pc_downsample and self.pc_downsample > 1:
                    step_t = int(self.pc_downsample)
                    idx = torch.arange(0, z_clean_t.shape[0], step_t, device=device)
                    z_clean_t = z_clean_t[idx]
                    u_clean_t = u_clean_t[idx]
                    v_clean_t = v_clean_t[idx]

                # Convert to 3D coordinates (using cached tensors)
                x_clean_t = (u_clean_t - self.cx_t) * z_clean_t / self.fx_t
                y_clean_t = (v_clean_t - self.cy_t) * z_clean_t / self.fy_t

                class_id = int(clss[i])
                instance_id = int(ids[i])

                r, g, b = self.get_color_for_class(str(class_id))
                rgb_packed = self.pack_rgb(r, g, b)

                # Build pointcloud using GPU tensors for speed
                N = x_clean_t.shape[0]
                if N == 0:
                    continue

                rgb_packed_t = torch.full((N,), float(rgb_packed), dtype=torch.float32, device=device)
                class_id_t = torch.full((N,), float(class_id), dtype=torch.float32, device=device)
                instance_id_t = torch.full((N,), float(instance_id), dtype=torch.float32, device=device)

                instance_cloud_t = torch.stack(
                    [
                        x_clean_t.to(torch.float32),
                        y_clean_t.to(torch.float32),
                        z_clean_t.to(torch.float32),
                        rgb_packed_t,
                        class_id_t,
                        instance_id_t,
                    ],
                    dim=1,
                )

                class_name = self.class_id_to_name(class_id)

                # Centroid computation on GPU (faster than numpy)
                centroid_x = float(torch.mean(x_clean_t).item())
                centroid_y = float(torch.mean(y_clean_t).item())
                centroid_z = float(torch.mean(z_clean_t).item())
                self.last_centroids.append({
                    "class_id": class_id,
                    "instance_id": instance_id,
                    "centroid": (centroid_x, centroid_y, centroid_z)
                })

                timestamp = Time(
                    sec=rgb_msg.header.stamp.sec,
                    nanosec=rgb_msg.header.stamp.nanosec
                )

                # Store per-detection metadata
                self.last_detection_meta.append({
                    "name": class_name,
                    "instance_id": instance_id,
                    "timestamp": timestamp,
                })
                print(f"Detected {class_name} (inst {instance_id}) at ({centroid_x:.3f}, {centroid_y:.3f}, {centroid_z:.3f})")

                # Process CLIP embeddings for this detection
                if clip_crop_info is not None:
                    sx1, sy1, sx2, sy2 = clip_crop_info["sx1"], clip_crop_info["sy1"], clip_crop_info["sx2"], clip_crop_info["sy2"]
                    x1, y1, x2, y2 = clip_crop_info["x1"], clip_crop_info["y1"], clip_crop_info["x2"], clip_crop_info["y2"]
                    crop_bgr = frame_bgr[sy1:sy2, sx1:sx2]
                    try:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        pil_crop = PILImage.fromarray(crop_rgb)
                        image_in = self.preprocess(pil_crop).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            feat = self.model2.encode_image(image_in)
                            feat = feat / feat.norm(dim=-1, keepdim=True)
                        self.last_clip_embeddings.append({
                            "class_id": clip_crop_info["class_id"],
                            "instance_id": clip_crop_info["instance_id"],
                            "bbox_full": [int(x1), int(y1), int(x2), int(y2)],
                            "square_crop": [int(sx1), int(sy1), int(sx2), int(sy2)],
                            "embedding": feat.squeeze(0).detach().float().cpu().numpy(),
                        })
                        
                        # Only create visualization if debug mode is enabled
                        if self.debug_clip_boxes:
                            if clip_boxes_vis is None:
                                clip_boxes_vis = frame_bgr.copy()
                            cv2.rectangle(clip_boxes_vis, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                            cv2.rectangle(clip_boxes_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            label = f"ID:{instance_id} cls:{class_id}"
                            cv2.putText(clip_boxes_vis, label, (sx1, sy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    except Exception as e:
                        self.get_logger().warn(f"CLIP embedding failed for det {i}: {e}")

                all_points_list.append(instance_cloud_t)

            if all_points_list:
                final_points_t = torch.cat(all_points_list, dim=0)
                final_points = final_points_t.detach().cpu().numpy().astype(np.float32)
                
                # Pointcloud computation timing (before publish)
                pc_end = self.get_clock().now()
                pc_time = (pc_end - pc_start).nanoseconds / 1e9
                self.get_logger().info(f"Pointcloud computation time: {pc_time:.3f} seconds, {1.0/pc_time:.2f} FPS")
                
                self.publish_pointcloud(final_points, depth_msg.header)

            try:
                annotated_bgr = res.plot()
                self.publish_annotated_image(annotated_bgr, rgb_msg.header)
            except Exception as e:
                self.get_logger().warn(f"Annotated publish failed: {e}")
            
            # Publish CLIP boxes visualization (only if we had detections with crops)
            if clip_boxes_vis is not None:
                self.publish_clip_boxes_image(clip_boxes_vis, rgb_msg.header)

            # Publish centroids via helper function for this frame
            self.publish_centroids(rgb_msg.header.stamp, depth_msg.header.frame_id)

            # Publish detected objects
            self.publish_detections(rgb_msg.header.stamp, depth_msg.header.frame_id)

            # End-to-end timing
            cb_end = self.get_clock().now()
            total_time = (cb_end - cb_start).nanoseconds / 1e9
            self.get_logger().info(f"End-to-end processing time: {total_time:.3f} seconds, {1.0/total_time:.2f} FPS")

        except Exception as e:
            self.get_logger().error(f"Error in synced_cb_vectorized: {e}")

    def create_centroid_marker(self, class_name: str, centroid: Vector3, class_id: int, marker_id: int, frame_id: str, stamp: Time):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "yolo_centroids"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(centroid.x)
        marker.pose.position.y = float(centroid.y)
        marker.pose.position.z = float(centroid.z)
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        cr, cg, cb = self.get_color_for_class(str(class_id))
        marker.color.r = float(cr) / 255.0
        marker.color.g = float(cg) / 255.0
        marker.color.b = float(cb) / 255.0
        marker.color.a = 0.9

        return marker

    def publish_centroids(self, stamp: Time, frame_id: str):
        """
        Publish visualization markers for the centroids detected in the current frame.
        Uses self.last_centroids which is populated during synced_cb.
        """
        try:
            if not self.last_centroids:
                return

            marker_array = MarkerArray()

            for idx, entry in enumerate(self.last_centroids):
                cx, cy, cz = entry["centroid"]
                class_id = int(entry["class_id"])
                class_name = self.class_id_to_name(class_id)
                centroid_vec = Vector3(x=float(cx), y=float(cy), z=float(cz))

                marker = self.create_centroid_marker(
                    class_name=class_name,
                    centroid=centroid_vec,
                    class_id=class_id,
                    marker_id=idx,
                    frame_id=frame_id,
                    stamp=stamp,
                )
                marker_array.markers.append(marker)

            self.marker_pub.publish(marker_array)
        except Exception as e:
            self.get_logger().warn(f"Centroid markers publish failed: {e}")

    def publish_pointcloud(self, points: np.ndarray, header):
        """Publish a PointCloud2 from Nx6 numpy array [x,y,z,rgb,class_id,instance_id]."""
        try:
            if points is None or points.size == 0:
                return
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name="class_id", offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name="instance_id", offset=20, datatype=PointField.FLOAT32, count=1),
            ]
            cloud_msg = point_cloud2.create_cloud(header, fields, points.tolist())
            self.pc_pub.publish(cloud_msg)
        except Exception as e:
            self.get_logger().warn(f"PointCloud publish failed: {e}")

    def publish_annotated_image(self, bgr_image: np.ndarray, header):
        """Publish an annotated image given BGR array and header."""
        try:
            anno_msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding="bgr8")
            anno_msg.header = header
            self.anno_pub.publish(anno_msg)
        except Exception as e:
            self.get_logger().warn(f"Annotated publish failed: {e}")

    def publish_clip_boxes_image(self, bgr_image: np.ndarray, header):
        """Publish the CLIP boxes visualization image given BGR array and header."""
        try:
            clip_msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding="bgr8")
            clip_msg.header = header
            self.clip_boxes_pub.publish(clip_msg)
        except Exception as e:
            self.get_logger().warn(f"CLIP boxes publish failed: {e}")

    def publish_detections(self, stamp: Time, frame_id: str):
        """Publish detected objects as custom messages."""
        try:
            if not self.last_detection_meta:
                return

            for meta in self.last_detection_meta:
                instance_id = meta["instance_id"]
                
                # Find matching centroid - skip if not found
                centroid_found = False
                for centroid_entry in self.last_centroids:
                    if centroid_entry["instance_id"] == instance_id:
                        cx, cy, cz = centroid_entry["centroid"]
                        centroid_vec = Vector3(x=float(cx), y=float(cy), z=float(cz))
                        centroid_found = True
                        break
                
                if not centroid_found:
                    self.get_logger().warn(f"Centroid not found for instance {instance_id}")
                    continue
                
                # Find matching CLIP embedding - use empty array if not found
                embedding_array = []
                for clip_entry in self.last_clip_embeddings:
                    if clip_entry["instance_id"] == instance_id:
                        embedding_array = clip_entry["embedding"].tolist()
                        break
                
                # Create and populate message
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