
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

class Yolo11SegNode(Node):
    def __init__(self):
        super().__init__("yolo11_seg_node")

        # ============= Parameters ============= #
        self.declare_parameter("model_path", "/home/sensor/yolo11n-seg.engine")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("pointcloud_topic", "/yolo/pointcloud")
        self.declare_parameter("annotated_topic", "/yolo/annotated")
        self.declare_parameter("clip_boxes_topic", "/yolo/clip_boxes")
        self.declare_parameter("detections_topic", "/yolo/detections")
        
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.70)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("retina_masks", True)
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("pc_downsample", 2)
        self.declare_parameter("pc_max_range", 8.0)
        self.declare_parameter("mask_threshold", 0.5)
        self.declare_parameter("clip_square_scale", 1.4)

        model_path = self.get_parameter("model_path").value
        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.pc_topic = self.get_parameter("pointcloud_topic").value
        self.anno_topic = self.get_parameter("annotated_topic").value
        self.clip_boxes_topic = self.get_parameter("clip_boxes_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value

        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.retina_masks = bool(self.get_parameter("retina_masks").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.pc_downsample = int(self.get_parameter("pc_downsample").value)
        self.pc_max_range = float(self.get_parameter("pc_max_range").value)
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        self.clip_square_scale = float(self.get_parameter("clip_square_scale").value)

        # ============= Initialization ============= #

        self.get_logger().info(f"Loading model: {model_path}")
        self.model = YOLO(model_path, task="segment")
        self.names = getattr(self.model, "names", {})

        qos_sensor = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

        self.bridge = CvBridge()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on {self.device}...")
        self.model2, self.preprocess = clip.load("ViT-B/32", device=self.device)
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
        self.class_colors = {}

        self.get_logger().info(f"Ready. Publishing to {self.pc_topic}")

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
            
            # Check timestamp synchronization (within 50ms)
            rgb_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            depth_time = self.latest_depth_msg.header.stamp.sec + self.latest_depth_msg.header.stamp.nanosec * 1e-9
            time_diff = abs(rgb_time - depth_time)
            
            if time_diff > 0.05:  # 50ms threshold
                self.get_logger().warn(f"RGB/Depth timestamp mismatch: {time_diff:.3f}s")
                return
            
            # Copy messages to avoid holding lock during processing
            rgb_msg = msg
            depth_msg = self.latest_depth_msg
        
        # Process outside of lock to avoid blocking callbacks
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

            # Convert ROS image messages into NumPy/OpenCV images
            frame_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            if depth_img.ndim != 2 or self.fx is None:
                return

            height, width = depth_img.shape

            # YOLOv11-seg model inference on ROI only
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

            # Process the first result
            res = results[0]

            if not hasattr(res, "boxes") or len(res.boxes) == 0: # No detections
                return

            xyxy = res.boxes.xyxy.cpu().numpy().astype(int) # Run the downstream code on CPU (ROI coords)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids  = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.zeros(len(clss))
            masks = res.masks.data.cpu().numpy() if hasattr(res, "masks") and res.masks is not None else None  # (N, roi_h, roi_w)

            all_points_list = []

            step = max(self.pc_downsample, 1)
            DEPTH_TOLERANCE = 0.5
            MIN_POINTS = 10

            DYNAMIC_CLASSES = {0, 1, 2, 3, 5, 7, 15, 16}
            
            is_uint16 = (depth_msg.encoding == "16UC1") # If depth is in uint16 format it is in mm
            scale_factor = (1.0 / self.depth_scale) if is_uint16 else 1.0 # Convert from mm to m

            # Reset CLIP embeddings list for this frame
            self.last_clip_embeddings = []
            # Reset centroids list for this frame
            self.last_centroids = []
            # Reset detection meta list for this frame
            self.last_detection_meta = []
            
            # Create visualization image for CLIP boxes
            clip_boxes_vis = frame_bgr.copy()

            for i in range(len(xyxy)): # Iterate over detections (ROI coords)
                
                class_id = int(clss[i])

                # Skip to next if dynamic class for point cloud generation
                if class_id in DYNAMIC_CLASSES:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))

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
                sx2 = max(0, min(sx2, width))
                sy2 = max(0, min(sy2, height))

                # CLIP embedding for the square crop (if valid)
                if sx2 > sx1 and sy2 > sy1:
                    crop_bgr = frame_bgr[sy1:sy2, sx1:sx2]
                    try:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        pil_crop = PILImage.fromarray(crop_rgb)
                        image_in = self.preprocess(pil_crop).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            feat = self.model2.encode_image(image_in)
                            feat = feat / feat.norm(dim=-1, keepdim=True)
                        # store on CPU to keep memory stable
                        self.last_clip_embeddings.append({
                            "class_id": class_id,
                            "instance_id": int(ids[i]) if isinstance(ids, np.ndarray) else 0,
                            "bbox_full": [int(x1), int(y1), int(x2), int(y2)],
                            "square_crop": [int(sx1), int(sy1), int(sx2), int(sy2)],
                            "embedding": feat.squeeze(0).detach().float().cpu().numpy(),
                        })
                        
                        # Draw CLIP square box on visualization (green for CLIP box)
                        cv2.rectangle(clip_boxes_vis, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                        # Draw original bbox (red)
                        cv2.rectangle(clip_boxes_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        # Add label
                        label = f"ID:{int(ids[i])} cls:{class_id}"
                        cv2.putText(clip_boxes_vis, label, (sx1, sy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    except Exception as e:
                        self.get_logger().warn(f"CLIP embedding failed for det {i}: {e}")

                if x2 <= x1 or y2 <= y1:
                    continue # Invalid / empty box

                valid_mask = (depth_img > 0) & (~np.isnan(depth_img)) # Remove invalid depth values
 
                if self.pc_max_range > 0.0:
                    valid_mask &= (depth_img * scale_factor <= self.pc_max_range)

                # Object support mask: segmentation if available, else rectangular bbox
                if masks is not None and i < len(masks):
                    obj_mask = masks[i] >= self.mask_threshold # shape (H, W)
                else:
                    continue

                # Pixels that are both depth-valid and belong to this object
                valid = valid_mask & obj_mask
                v_coords, u_coords = np.where(valid) # Pixel coordinates of valid points

                # Require minimum number of points
                if v_coords.size < MIN_POINTS:
                    continue

                z_vals = depth_img[v_coords, u_coords].astype(np.float32) * scale_factor # Depth values in meters

                # Robust filtering around median depth
                median_z = np.median(z_vals)
                keep_mask = np.abs(z_vals - median_z) < DEPTH_TOLERANCE
                
                if not np.any(keep_mask):
                    continue

                z_clean = z_vals[keep_mask]
                u_clean = u_coords[keep_mask]
                v_clean = v_coords[keep_mask]

                x_clean = (u_clean - self.cx) * z_clean / self.fx
                y_clean = (v_clean - self.cy) * z_clean / self.fy

                class_id = int(clss[i])
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

                class_name = self.names[class_id]

                # Centroid computation (mean of filtered 3D points)
                centroid_x = float(np.mean(x_clean))
                centroid_y = float(np.mean(y_clean))
                centroid_z = float(np.mean(z_clean))
                self.last_centroids.append({
                    "class_id": class_id,
                    "instance_id": instance_id,
                    "centroid": (centroid_x, centroid_y, centroid_z)
                })

                timestamp = Time(
                    sec=rgb_msg.header.stamp.sec,
                    nanosec=rgb_msg.header.stamp.nanosec
                )

                # Store per-detection metadata (not used elsewhere for now)
                self.last_detection_meta.append({
                    "name": class_name,
                    "instance_id": instance_id,
                    "timestamp": timestamp,
                })

                # (Marker creation moved to publish_centroids for clarity)
                
                all_points_list.append(instance_cloud)

            if all_points_list:
                final_points = np.vstack(all_points_list)
                self.publish_pointcloud(final_points, depth_msg.header)

            try:
                annotated_bgr = res.plot()
                self.publish_annotated_image(annotated_bgr, rgb_msg.header)
            except Exception as e:
                self.get_logger().warn(f"Annotated publish failed: {e}")
            
            # Publish CLIP boxes visualization
            self.publish_clip_boxes_image(clip_boxes_vis, rgb_msg.header)

            # Publish centroids via helper function for this frame
            self.publish_centroids(rgb_msg.header.stamp, depth_msg.header.frame_id)

            # Publish detected objects
            self.publish_detections(rgb_msg.header.stamp, depth_msg.header.frame_id)

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
                class_name = self.names[class_id] if class_id in self.names else str(class_id)
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