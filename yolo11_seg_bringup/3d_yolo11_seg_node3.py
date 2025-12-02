#!/usr/bin/env python3
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
import cv2

class Yolo11SegNode(Node):
    def __init__(self):
        super().__init__("yolo11_seg_node")

        # ============= Parameters ============= #
        self.declare_parameter("model_path", "/home/sensor/yolo11n-seg.engine")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("pointcloud_topic", "/yolo/pointcloud")
        
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.70)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("retina_masks", True)
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("pc_downsample", 2)
        self.declare_parameter("pc_max_range", 8.0)
        self.declare_parameter("mask_threshold", 0.5)

        model_path = self.get_parameter("model_path").value
        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.pc_topic = self.get_parameter("pointcloud_topic").value

        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.retina_masks = bool(self.get_parameter("retina_masks").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.pc_downsample = int(self.get_parameter("pc_downsample").value)
        self.pc_max_range = float(self.get_parameter("pc_max_range").value)
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)

        # ============= Initialization ============= #

        self.get_logger().info(f"Loading model: {model_path}")
        self.model = YOLO(model_path, task="segment")
        self.names = getattr(self.model, "names", {})

        qos_sensor = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

        self.bridge = CvBridge()
 
        self.latest_depth_msg = None
        self.sync_lock = threading.Lock()

        self.rgb_sub = self.create_subscription(Image, self.image_topic, self.rgb_callback, qos_profile=qos_sensor)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile=qos_sensor)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_profile=qos_sensor)

        self.pc_pub = self.create_publisher(PointCloud2, self.pc_topic, 10)

        self.fx = self.fy = self.cx = self.cy = None
        self.class_colors = {}

        # ============= Rate Tracking ============= #
        self.inference_count = 0
        self.inference_start_time = self.get_clock().now()
        self.publish_count = 0
        self.publish_start_time = self.get_clock().now()

        self.get_logger().info(f"Ready. Publishing to {self.pc_topic}")

    def camera_info_cb(self, msg: CameraInfo):
        """Read camera intrinsic parameters."""
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

    def get_color_for_class(self, class_id: str):
        """Deterministically map a class_id to an RGB color."""
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
    def pack_rgb(r: int, g: int, b: int) -> float:
        """Pack 3x uint8 RGB into float32 for PointCloud2 'rgb' field."""
        rgb_uint32 = (r << 16) | (g << 8) | b
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]

    def synced_cb(self, rgb_msg, depth_msg):
        """
        Perform YOLOv11-seg inference and generate colored 3D point cloud.
        """
        try:
            # Convert ROS images to OpenCV
            frame_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            if depth_img.ndim != 2 or self.fx is None:
                return

            height, width = depth_img.shape

            # YOLO inference
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

            # Update inference rate tracking
            self.inference_count += 1
            elapsed = (self.get_clock().now() - self.inference_start_time).nanoseconds / 1e9
            if elapsed >= 1.0:  # Log rate every second
                rate = self.inference_count / elapsed
                self.get_logger().info(f"YOLO inference rate: {rate:.2f} Hz")
                self.inference_count = 0
                self.inference_start_time = self.get_clock().now()

            res = results[0]

            if not hasattr(res, "boxes") or len(res.boxes) == 0:
                return

            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids  = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.zeros(len(clss))
            masks = res.masks.data.cpu().numpy() if hasattr(res, "masks") and res.masks is not None else None

            all_points_list = []

            DEPTH_TOLERANCE = 0.5
            MIN_POINTS = 10
            DYNAMIC_CLASSES = {0, 1, 2, 3, 5, 7, 15, 16}
            
            is_uint16 = (depth_msg.encoding == "16UC1")
            scale_factor = (1.0 / self.depth_scale) if is_uint16 else 1.0

            for i in range(len(xyxy)):
                class_id = int(clss[i])

                # Skip dynamic classes
                if class_id in DYNAMIC_CLASSES:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                valid_mask = (depth_img > 0) & (~np.isnan(depth_img))
 
                if self.pc_max_range > 0.0:
                    valid_mask &= (depth_img * scale_factor <= self.pc_max_range)

                # Use segmentation mask if available
                if masks is not None and i < len(masks):
                    obj_mask = masks[i] >= self.mask_threshold
                else:
                    continue

                valid = valid_mask & obj_mask
                v_coords, u_coords = np.where(valid)

                if v_coords.size < MIN_POINTS:
                    continue

                z_vals = depth_img[v_coords, u_coords].astype(np.float32) * scale_factor

                # Filter around median depth
                median_z = np.median(z_vals)
                keep_mask = np.abs(z_vals - median_z) < DEPTH_TOLERANCE
                
                if not np.any(keep_mask):
                    continue

                z_clean = z_vals[keep_mask]
                u_clean = u_coords[keep_mask]
                v_clean = v_coords[keep_mask]

                # Convert to 3D coordinates
                x_clean = (u_clean - self.cx) * z_clean / self.fx
                y_clean = (v_clean - self.cy) * z_clean / self.fy

                instance_id = int(ids[i])

                r, g, b = self.get_color_for_class(str(class_id))
                rgb_packed = self.pack_rgb(r, g, b)

                N = x_clean.size
                instance_cloud = np.zeros((N, 6), dtype=np.float32)
                instance_cloud[:, 0] = x_clean
                instance_cloud[:, 1] = y_clean
                instance_cloud[:, 2] = z_clean
                instance_cloud[:, 3] = rgb_packed
                instance_cloud[:, 4] = class_id
                instance_cloud[:, 5] = instance_id

                all_points_list.append(instance_cloud)

            if all_points_list:
                final_points = np.vstack(all_points_list)
                self.publish_pointcloud(final_points, depth_msg.header)
                
                # Update pointcloud publishing rate tracking
                self.publish_count += 1
                elapsed = (self.get_clock().now() - self.publish_start_time).nanoseconds / 1e9
                if elapsed >= 1.0:  # Log rate every second
                    rate = self.publish_count / elapsed
                    self.get_logger().info(f"PointCloud publishing rate: {rate:.2f} Hz")
                    self.publish_count = 0
                    self.publish_start_time = self.get_clock().now()

        except Exception as e:
            self.get_logger().error(f"Error in synced_cb: {e}")

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