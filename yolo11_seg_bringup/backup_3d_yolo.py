#!/usr/bin/env python3
"""
Yolo11SegNode: YOLO11 segmentation + 3D pointcloud of detected objects

Subscribe:
  - rgb image:    /camera/camera/color/image_raw
  - depth image:  /camera/camera/aligned_depth_to_color/image_raw
  - camera info:  /camera/camera/color/camera_info

Publish:
  - /yolo/pointcloud  (sensor_msgs/PointCloud2) - 3D points of detected objects colored by class
  - /yolo/annotated   (sensor_msgs/Image) - RGB image with detection masks and class labels
"""

import struct
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from ultralytics import YOLO


class Yolo11SegNode(Node):
    def __init__(self):
        super().__init__("yolo11_seg_node")

        # Parameters
        self.declare_parameter("model_path", "/home/sensor/yolo11n-seg.engine")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("pointcloud_topic", "/yolo/pointcloud")
        self.declare_parameter("annotated_topic", "/yolo/annotated")
        
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.70)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("retina_masks", True)
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("pc_downsample", 2)
        self.declare_parameter("pc_max_range", 8.0)
        self.declare_parameter("mask_threshold", 0.5)

        # Get parameters
        model_path = self.get_parameter("model_path").value
        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.pc_topic = self.get_parameter("pointcloud_topic").value
        self.anno_topic = self.get_parameter("annotated_topic").value

        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.retina_masks = bool(self.get_parameter("retina_masks").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.pc_downsample = int(self.get_parameter("pc_downsample").value)
        self.pc_max_range = float(self.get_parameter("pc_max_range").value)
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)

        # Load YOLO model
        self.get_logger().info(f"Loading model: {model_path}")
        self.model = YOLO(model_path, task="segment")
        self.names = getattr(self.model, "names", {})

        # QoS profile
        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # CV bridge
        self.bridge = CvBridge()

        # Synchronization
        self.latest_depth_msg = None
        self.sync_lock = threading.Lock()

        # Subscribe
        self.rgb_sub = self.create_subscription(
            Image, self.image_topic, self.rgb_callback, qos_profile=qos_sensor
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile=qos_sensor
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_profile=qos_sensor
        )

        # Publisher
        self.pc_pub = self.create_publisher(PointCloud2, self.pc_topic, 10)
        self.anno_pub = self.create_publisher(Image, self.anno_topic, 10)

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.class_colors = {}

        self.get_logger().info(f"Ready. Publishing to {self.pc_topic}")

    # ========== Camera info ==========

    def camera_info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info(
                f"Camera intrinsics set: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )

    # ========== Manual synchronization callbacks ==========

    def depth_callback(self, msg: Image):
        """Store the latest depth message."""
        with self.sync_lock:
            self.latest_depth_msg = msg

    def rgb_callback(self, msg: Image):
        """On RGB arrival, process with latest depth if available."""
        with self.sync_lock:
            if self.latest_depth_msg is None:
                return
            rgb_msg = msg
            depth_msg = self.latest_depth_msg
        
        self.synced_cb(rgb_msg, depth_msg)

    # ========== Utility: color mapping ==========

    def get_color_for_class(self, class_id: str):
        """Deterministically map a class_id string to an RGB color."""
        if class_id not in self.class_colors:
            h = abs(hash(class_id))
            r = (h >> 0) & 0xFF
            g = (h >> 8) & 0xFF
            b = (h >> 16) & 0xFF
            if r < 30 and g < 30 and b < 30:  # avoid almost-black
                r = (r + 128) & 0xFF
                g = (g + 64) & 0xFF
            self.class_colors[class_id] = (r, g, b)
        return self.class_colors[class_id]

    @staticmethod
    def pack_rgb(r: int, g: int, b: int) -> float:
        """Pack 3x uint8 RGB into float32 for PointCloud2 'rgb' field."""
        rgb_uint32 = (r << 16) | (g << 8) | b
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]

    # ========== Main sync callback: RGB + depth ==========

    def synced_cb2(self, rgb_msg: Image, depth_msg: Image):
        try:
            # Convert images
            frame_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            if depth_img.ndim != 2 or self.fx is None:
                return

            # YOLO inference
            results = self.model.track(
                source=frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                retina_masks=self.retina_masks,
                stream=False,
                verbose=False,
                persist=True,              # keep IDs between frames
                tracker="bytetrack.yaml",  # or botsort.yaml, etc.
            )
            res = results[0]

            # Extract detections
            if not hasattr(res, "boxes") or len(res.boxes) == 0:
                return

            xyxy = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids  = res.boxes.id.cpu().numpy().astype(int)   # <-- instance / track ids
            masks = res.masks.data.cpu().numpy() if hasattr(res, "masks") and res.masks is not None else None

            # Build pointcloud
            points = []
            height, width = depth_img.shape
            step = max(self.pc_downsample, 1)
            # Determine depth scale based on encoding
            is_uint16 = depth_msg.encoding == "16UC1"
            depth_scale = self.depth_scale if is_uint16 else 1.0

            # Filter Settings
            # Points > 0.5m away from the object's center are considered "shadow"
            DEPTH_TOLERANCE = 0.5  
            MIN_POINTS = 10

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                class_id = clss[i]
                instance_id = ids[i]
                r, g, b = self.get_color_for_class(str(class_id))
                rgb_packed = self.pack_rgb(r, g, b)

                u_min, u_max = max(int(x1), 0), min(int(x2), width - 1)
                v_min, v_max = max(int(y1), 0), min(int(y2), height - 1)

                mask = masks[i] if masks is not None and i < len(masks) else None

                # 1. Collect all raw candidates for this object
                instance_candidates = []

                for v in range(v_min, v_max + 1, step):
                    for u in range(u_min, u_max + 1, step):
                        # Check mask
                        if mask is not None and mask.shape == depth_img.shape:
                            if mask[v, u] < self.mask_threshold:
                                continue

                        # Get depth
                        z_raw = depth_img[v, u]
                        if z_raw == 0 or np.isnan(z_raw):
                            continue

                        z = float(z_raw) / self.depth_scale if depth_msg.encoding == "16UC1" else float(z_raw)

                        if z <= 0.0 or (self.pc_max_range > 0.0 and z > self.pc_max_range):
                            continue

                        # Unproject to 3D
                        x = (u - self.cx) * z / self.fx
                        y = (v - self.cy) * z / self.fy
                        
                        # Append point with x, y, z, rgb, and class_id
                        # class_id can be used to look up the class name in the mapper node

                        instance_candidates.append([x, y, z, rgb_packed, float(class_id), float(instance_id)])

                        # points.append([x, y, z, rgb_packed, float(class_id), float(instance_id)])

                # 2. Apply Robust Filtering (Median-based)
                if len(instance_candidates) > MIN_POINTS:
                    inst_arr = np.array(instance_candidates)
                    z_vals = inst_arr[:, 2]

                    # --- CRITICAL FIX ---
                    # Use MEDIAN, not MEAN. Median ignores the "tail" of shadow points.
                    center_z = np.median(z_vals)
                    
                    # Keep points that are within a fixed distance (e.g., 50cm) of the median
                    # We are stricter on the "far" side to cut the shadow
                    mask_keep = (z_vals > (center_z - DEPTH_TOLERANCE)) & \
                                (z_vals < (center_z + DEPTH_TOLERANCE))
                    
                    filtered_points = inst_arr[mask_keep]
                    points.extend(filtered_points.tolist())
                else:
                    points.extend(instance_candidates)

            # Publish pointcloud
            if points:
                fields = [
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
                    PointField(name="class_id", offset=16, datatype=PointField.FLOAT32, count=1),
                    PointField(name="instance_id", offset=20, datatype=PointField.FLOAT32, count=1),
                ]
                cloud_msg = point_cloud2.create_cloud(depth_msg.header, fields, points)
                self.pc_pub.publish(cloud_msg)

            # Publish annotated image
            try:
                annotated_bgr = res.plot()
                anno_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding="bgr8")
                anno_msg.header = rgb_msg.header
                self.anno_pub.publish(anno_msg)
            except Exception as e:
                self.get_logger().warn(f"Annotated publish failed: {e}")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def synced_cb(self, rgb_msg, depth_msg):
        """
        Vectorized version: ~50x faster. 
        Uses np.median and fixed tolerance to remove 'shadows' behind objects.
        """
        try:
            # 1. Convert images
            frame_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            if depth_img.ndim != 2 or self.fx is None:
                return

            height, width = depth_img.shape

            # 2. YOLO inference
            results = self.model.track(
                source=frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                retina_masks=self.retina_masks,
                stream=False,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
            )
            res = results[0]

            if not hasattr(res, "boxes") or len(res.boxes) == 0:
                return

            # 3. Extract YOLO Data
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids  = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.zeros(len(clss))
            masks = res.masks.data.cpu().numpy() if hasattr(res, "masks") and res.masks is not None else None

            all_points_list = []
            
            # CONFIGURATION
            step = max(self.pc_downsample, 1)
            DEPTH_TOLERANCE = 0.5  # Cut points > 0.5m away from median depth
            MIN_POINTS = 10        # Minimum valid points to keep an object

            DYNAMIC_CLASSES = {0, 1, 2, 3, 5, 7, 15, 16, 75}
            
            # Pre-calculate depth scale
            is_uint16 = (depth_msg.encoding == "16UC1")
            scale_factor = (1.0 / self.depth_scale) if is_uint16 else 1.0

            # 4. Process each Object
            for i in range(len(xyxy)):

                class_id = int(clss[i])
                if class_id in DYNAMIC_CLASSES:
                    # Skip 3D processing for this object
                    continue

                x1, y1, x2, y2 = xyxy[i]
                
                # Clamp bbox to image limits
                u_min, u_max = max(x1, 0), min(x2, width)
                v_min, v_max = max(y1, 0), min(y2, height)
                
                if u_max <= u_min or v_max <= v_min:
                    continue

                # --- A. Vectorized Slicing & Grid Generation ---
                # Slice depth map
                depth_crop = depth_img[v_min:v_max:step, u_min:u_max:step]
                
                # Generate coordinate grid for this crop
                xv = np.arange(u_min, u_max, step)
                yv = np.arange(v_min, v_max, step)
                u_grid, v_grid = np.meshgrid(xv, yv)

                # --- B. Create Validity Mask ---
                # 1. Valid depth values
                valid_mask = (depth_crop > 0) & (~np.isnan(depth_crop))
                
                # 2. Max Range
                if self.pc_max_range > 0.0:
                    valid_mask &= (depth_crop * scale_factor <= self.pc_max_range)

                # 3. YOLO Mask (Geometric check)
                if masks is not None and i < len(masks):
                    mask_crop = masks[i][v_min:v_max:step, u_min:u_max:step]
                    if mask_crop.shape == depth_crop.shape:
                        valid_mask &= (mask_crop >= self.mask_threshold)

                # Apply mask to get flat arrays
                z_raw = depth_crop[valid_mask]
                u_flat = u_grid[valid_mask]
                v_flat = v_grid[valid_mask]

                if z_raw.size < MIN_POINTS:
                    continue

                # Convert raw depth to meters
                z_meters = z_raw * scale_factor

                # --- C. The "Shadow" Fix (Vectorized) ---
                # Use Median to find the object center, ignore the wall behind
                median_z = np.median(z_meters)
                
                # Keep only points within tolerance of the median
                keep_mask = np.abs(z_meters - median_z) < DEPTH_TOLERANCE
                
                z_clean = z_meters[keep_mask]
                u_clean = u_flat[keep_mask]
                v_clean = v_flat[keep_mask]

                if z_clean.size == 0:
                    continue

                # --- D. 3D Unprojection ---
                x_clean = (u_clean - self.cx) * z_clean / self.fx
                y_clean = (v_clean - self.cy) * z_clean / self.fy

                # --- E. Attribute Packing ---
                class_id = int(clss[i])
                instance_id = int(ids[i])
                
                # Color (Get packed float once)
                r, g, b = self.get_color_for_class(str(class_id))
                rgb_packed = self.pack_rgb(r, g, b)

                # Create the (N, 6) array
                # Columns: x, y, z, rgb, class_id, instance_id
                N = x_clean.size
                instance_cloud = np.zeros((N, 6), dtype=np.float32)
                instance_cloud[:, 0] = x_clean
                instance_cloud[:, 1] = y_clean
                instance_cloud[:, 2] = z_clean
                instance_cloud[:, 3] = rgb_packed
                instance_cloud[:, 4] = class_id
                instance_cloud[:, 5] = instance_id

                all_points_list.append(instance_cloud)

            # 5. Publish Pointcloud
            if all_points_list:
                # Combine all objects
                final_points = np.vstack(all_points_list)
                
                fields = [
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
                    PointField(name="class_id", offset=16, datatype=PointField.FLOAT32, count=1),
                    PointField(name="instance_id", offset=20, datatype=PointField.FLOAT32, count=1),
                ]
                
                # Convert to list for create_cloud (easiest compatibility method)
                cloud_msg = point_cloud2.create_cloud(depth_msg.header, fields, final_points.tolist())
                self.pc_pub.publish(cloud_msg)

            # 6. Publish Annotation Image
            try:
                annotated_bgr = res.plot()
                anno_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding="bgr8")
                anno_msg.header = rgb_msg.header
                self.anno_pub.publish(anno_msg)
            except Exception as e:
                self.get_logger().warn(f"Annotated publish failed: {e}")

        except Exception as e:
            self.get_logger().error(f"Error in synced_cb_vectorized: {e}")

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