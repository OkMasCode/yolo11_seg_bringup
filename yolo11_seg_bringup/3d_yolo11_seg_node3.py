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
import torch
from message_filters import Subscriber, ApproximateTimeSynchronizer

class Yolo11SegNode(Node):
    def __init__(self):
        super().__init__("yolo11_seg_node")

        model_path = "/home/sensor/yolo11n-seg.engine"
        self.image_topic = "/camera/camera/color/image_raw"
        self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
        self.camera_info_topic = "/camera/camera/color/camera_info"
        self.pc_topic = "/yolo/pointcloud"

        self.conf = 0.25
        self.iou = 0.70
        self.imgsz = 640
        self.retina_masks = True
        self.depth_scale = 1000.0
        self.pc_downsample = 2
        self.pc_max_range = 8.0
        self.mask_threshold = 0.5

        self.get_logger().info(f"Loading model: {model_path}")
        self.model = YOLO(model_path, task="segment")
        self.names = getattr(self.model, "names", {})

        qos_sensor = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

        self.bridge = CvBridge()

        self.latest_depth_msg = None
        self.sync_lock = threading.Lock() # To protect access to latest_depth_msg

        self.rgb_sub = self.create_subscription(Image, self.image_topic, self.rgb_callback, qos_profile=qos_sensor)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile=qos_sensor)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_profile=qos_sensor)

        self.pc_pub = self.create_publisher(PointCloud2, self.pc_topic, 10)

        self.fx = self.fy = self.cx = self.cy = None
        self.fx_t = self.fy_t = self.cx_t = self.cy_t = None  # Cached tensors
        self.class_colors = {}

    def camera_info_cb(self, msg: CameraInfo):
        """Read camera intrinsic parameters."""
        if self.fx is None:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]

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
            # Start end-to-end timing
            cb_start = self.get_clock().now()
            
            # Convert ROS images to OpenCV
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
            yolo_end = self.get_clock().now()
            yolo_time = (yolo_end - yolo_start).nanoseconds / 1e9
            self.get_logger().info(f"YOLO inference time: {yolo_time:.3f} seconds, {1.0/yolo_time:.2f} FPS")

            res = results[0]

            if not hasattr(res, "boxes") or len(res.boxes) == 0:
                self.get_logger().info("No objects detected, skipping pointcloud", throttle_duration_sec=2.0)
                return

            # Start pointcloud computation timing
            pc_start = self.get_clock().now()
            
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids  = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.zeros(len(clss))
            masks_t = res.masks.data if hasattr(res, "masks") and res.masks is not None else None

            all_points_list = []

            DEPTH_TOLERANCE = 0.5
            MIN_POINTS = 10
            # DYNAMIC_CLASSES = {0, 1, 2, 3, 5, 7, 15, 16}
            
            is_uint16 = (depth_msg.encoding == "16UC1")
            scale_factor = (1.0 / self.depth_scale) if is_uint16 else 1.0

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

            for i in range(len(xyxy)):
                class_id = int(clss[i])

                # Skip dynamic classes
                # if class_id in DYNAMIC_CLASSES:
                #     continue

                x1, y1, x2, y2 = xyxy[i]
                
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Use segmentation mask if available
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
                    step = int(self.pc_downsample)
                    idx = torch.arange(0, z_clean_t.shape[0], step, device=device)
                    z_clean_t = z_clean_t[idx]
                    u_clean_t = u_clean_t[idx]
                    v_clean_t = v_clean_t[idx]

                # Convert to 3D coordinates (using cached tensors)
                x_clean_t = (u_clean_t - self.cx_t) * z_clean_t / self.fx_t
                y_clean_t = (v_clean_t - self.cy_t) * z_clean_t / self.fy_t

                instance_id = int(ids[i])

                r, g, b = self.get_color_for_class(str(class_id))
                rgb_packed = self.pack_rgb(r, g, b)

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

                all_points_list.append(instance_cloud_t)

            if all_points_list:
                final_points_t = torch.cat(all_points_list, dim=0)
                final_points = final_points_t.detach().cpu().numpy().astype(np.float32)
                
                # Pointcloud computation timing (before publish)
                pc_end = self.get_clock().now()
                pc_time = (pc_end - pc_start).nanoseconds / 1e9
                self.get_logger().info(f"Pointcloud computation time: {pc_time:.3f} seconds, {1.0/pc_time:.2f} FPS")
                
                self.publish_pointcloud(final_points, depth_msg.header)
                self.get_logger().info(f"Published pointcloud with {final_points.shape[0]} points", throttle_duration_sec=2.0)
                
                # End-to-end timing
                cb_end = self.get_clock().now()
                total_time = (cb_end - cb_start).nanoseconds / 1e9
                self.get_logger().info(f"End-to-end processing time: {total_time:.3f} seconds, {1.0/total_time:.2f} FPS")
            else:
                self.get_logger().info("No valid points to publish", throttle_duration_sec=2.0)

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