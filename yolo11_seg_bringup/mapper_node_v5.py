import threading
import traceback
import struct
import zlib

import cv2
import open3d as o3d
from cv_bridge import CvBridge
import message_filters
import rclpy
import tf2_ros
import numpy as np
from geometry_msgs.msg import Vector3
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

from yolo11_seg_bringup.mapper_v5 import SemanticObjectMapV5
from yolo11_seg_interfaces.msg import DetectedObjectV3, DetectedObjectV3Array, SemanticObject, SemanticObjectArray


class PointCloudMapperNodeV5(Node):
    """Minimal semantic mapper node with clear and tunable behavior."""

    def __init__(self):
        super().__init__('pointcloud_mapper_node_v5')

        self.get_logger().info('[mapper_node_v5:init] starting initialization')

        # Core I/O.
        self.declare_parameter('detection_message', '/vision/detections')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('output_dir', '/workspaces/ros2_ws/src/yolo11_seg_bringup/config/')
        self.declare_parameter('export_interval', 3.0)
        self.declare_parameter('input_map_file', 'map_v5.json')
        self.declare_parameter('output_map_file', 'map_v5.json')
        self.declare_parameter('stable_pointcloud_topic', '/vision/semantic_map_v5/points')
        self.declare_parameter('publish_stable_pointcloud', True)

        # False-positive suppression.
        self.declare_parameter('min_input_confidence', 0.55)
        self.declare_parameter('confirmation_min_hits', 5)
        self.declare_parameter('confirmation_min_age_sec', 1)
        self.declare_parameter('min_avg_confidence_for_promotion', 0.50)

        # Geometry gating.
        self.declare_parameter('min_detection_depth_m', 0.25)
        self.declare_parameter('max_detection_depth_m', 4.0)
        self.declare_parameter('min_association_iou', 0.08)
        self.declare_parameter('min_cross_class_iou', 0.20)
        self.declare_parameter('class_mismatch_penalty', 0.25)

        # Class consensus.
        self.declare_parameter('class_count_weight', 1.0)
        self.declare_parameter('class_confidence_weight', 2.0)
        self.declare_parameter('class_switch_margin', 0.75)
        self.declare_parameter('min_class_votes_to_lock', 4)

        # Large-object merge + small-object protection.
        self.declare_parameter('enable_detection_bbox_merge', True)
        self.declare_parameter('detection_bbox_merge_min_extent', 1.00)
        self.declare_parameter('small_object_max_extent', 1.00)
        self.declare_parameter('small_object_assoc_min_iou', 0.30)

        # Periodic per-object clustering cleanup.
        self.declare_parameter('enable_largest_cluster_filter', True)
        self.declare_parameter('cluster_filter_eps_m', 0.18)
        self.declare_parameter('cluster_filter_min_points', 12)

        self.dm_topic = self.get_parameter('detection_message').value
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_dir').value
        self.export_interval = float(self.get_parameter('export_interval').value)
        self.input_map_file = self.get_parameter('input_map_file').value
        self.output_map_file = self.get_parameter('output_map_file').value
        self.stable_pointcloud_topic = self.get_parameter('stable_pointcloud_topic').value
        self.publish_stable_pointcloud_enabled = bool(self.get_parameter('publish_stable_pointcloud').value)

        self.get_logger().info(
            "[mapper_node_v5:init:params] "
            f"topic={self.dm_topic} map_frame={self.map_frame} camera_frame={self.camera_frame} "
            f"output_dir={self.output_dir} export_interval={self.export_interval} "
            f"load_map_on_start={self.load_map_on_start} input_map={self.input_map_file} output_map={self.output_map_file} "
            f"stable_cloud_topic={self.stable_pointcloud_topic} publish_stable_cloud={self.publish_stable_pointcloud_enabled}"
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.semantic_map = SemanticObjectMapV5(self.tf_buffer, self)

        # Map simple node params into mapper internals.
        self.semantic_map.min_input_confidence = float(self.get_parameter('min_input_confidence').value)
        self.semantic_map.confirmation_min_hits = int(self.get_parameter('confirmation_min_hits').value)
        self.semantic_map.confirmation_min_age_sec = float(self.get_parameter('confirmation_min_age_sec').value)
        self.semantic_map.min_avg_confidence_for_promotion = float(
            self.get_parameter('min_avg_confidence_for_promotion').value
        )
        self.semantic_map.min_detection_depth_m = float(self.get_parameter('min_detection_depth_m').value)
        self.semantic_map.max_detection_depth_m = float(self.get_parameter('max_detection_depth_m').value)
        self.semantic_map.min_association_iou = float(self.get_parameter('min_association_iou').value)
        self.semantic_map.min_cross_class_iou = float(self.get_parameter('min_cross_class_iou').value)
        self.semantic_map.class_mismatch_penalty = float(self.get_parameter('class_mismatch_penalty').value)

        self.semantic_map.class_count_weight = float(self.get_parameter('class_count_weight').value)
        self.semantic_map.class_confidence_weight = float(self.get_parameter('class_confidence_weight').value)
        self.semantic_map.class_switch_margin = float(self.get_parameter('class_switch_margin').value)
        self.semantic_map.min_class_votes_to_lock = int(self.get_parameter('min_class_votes_to_lock').value)

        self.semantic_map.enable_detection_bbox_merge = bool(self.get_parameter('enable_detection_bbox_merge').value)
        self.semantic_map.detection_bbox_merge_min_extent = float(
            self.get_parameter('detection_bbox_merge_min_extent').value
        )
        self.semantic_map.small_object_max_extent = float(self.get_parameter('small_object_max_extent').value)
        self.semantic_map.small_object_assoc_min_iou = float(self.get_parameter('small_object_assoc_min_iou').value)
        self.semantic_map.enable_largest_cluster_filter = bool(
            self.get_parameter('enable_largest_cluster_filter').value
        )
        self.semantic_map.cluster_filter_eps_m = float(self.get_parameter('cluster_filter_eps_m').value)
        self.semantic_map.cluster_filter_min_points = int(self.get_parameter('cluster_filter_min_points').value)

        self.get_logger().info(
            "[mapper_node_v5:init:mapper_params] "
            f"min_input_conf={self.semantic_map.min_input_confidence} "
            f"confirm_hits={self.semantic_map.confirmation_min_hits} "
            f"confirm_min_age={self.semantic_map.confirmation_min_age_sec} "
            f"depth_range=({self.semantic_map.min_detection_depth_m},{self.semantic_map.max_detection_depth_m}) "
            f"assoc_iou={self.semantic_map.min_association_iou} "
            f"largest_cluster_filter={self.semantic_map.enable_largest_cluster_filter} "
            f"cluster_eps={self.semantic_map.cluster_filter_eps_m} "
            f"cluster_min_pts={self.semantic_map.cluster_filter_min_points}"
        )

        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None

        self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self.camera_info_cb, 10
        )

        self.mask_sub = message_filters.Subscriber(
            self, DetectedObjectV3Array, self.dm_topic, qos_profile=qos_sensor
        )

        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth', qos_profile=qos_sensor
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.mask_sub, self.depth_sub], queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.synced_detection_callback)

        self.map_pub = self.create_publisher(SemanticObjectArray, '/vision/semantic_map_v5', 10)
        self.stable_cloud_pub = self.create_publisher(PointCloud2, self.stable_pointcloud_topic, 10)
        self.export_timer = self.create_timer(self.export_interval, self.export_callback)

        self.lock = threading.Lock()

        self.get_logger().info(
            f"[mapper_node_v5] ready. input={self.dm_topic}, output_dir={self.output_dir}, export={self.output_map_file}"
        )

    def camera_info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(f"Mapper received intrinsics: fx={self.fx}, fy={self.fy}")

    def get_points_in_mask(self, depth_m, binary_mask) -> np.ndarray:
        """Projects a 2D boolean mask and Depth image into 3D Camera Coordinates."""
        v, u = np.where(binary_mask)
        if len(u) == 0:
            return np.empty((0, 3), dtype=np.float32)

        z = depth_m[v, u]
        
        # Valid depth gate (e.g., 0.25m to 4.0m)
        valid_z = np.isfinite(z) & (z >= 0.25) & (z <= 4.0)
        if not np.any(valid_z):
            return np.empty((0, 3), dtype=np.float32)

        u = u[valid_z].astype(np.float32)
        v = v[valid_z].astype(np.float32)
        z = z[valid_z].astype(np.float32)

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return np.stack([x, y, z], axis=-1)

    def synced_detection_callback(self, mask_msg: DetectedObjectV3Array, depth_msg: Image):
        if self.fx is None:
            return # Wait until we have camera intrinsics
            
        try:
            with self.lock:
                # 1. Convert ROS Depth to NumPy float32 (meters)
                depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth_m = depth_raw.astype(np.float32) / 1000.0 if depth_raw.dtype == np.uint16 else depth_raw.astype(np.float32)

                object_names = []
                tracker_ids = []
                embeddings_list = []
                confidences = []
                points_cam_list = []

                # 2. Iterate through batched 2D detections
                for det in mask_msg.detections:
                    if det.mask.width == 0 or det.mask.height == 0:
                        continue
                        
                    # A. Extract 2D mask
                    cv_mask = self.bridge.imgmsg_to_cv2(det.mask, desired_encoding="mono8")
                    
                    # Optional: Resize mask to match depth shape if sensors differ
                    if cv_mask.shape != depth_m.shape:
                        cv_mask = cv2.resize(cv_mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                    binary_mask = (cv_mask > 0)

                    # B. Project to 3D
                    pts_cam = self.get_points_in_mask(depth_m, binary_mask)
                    if len(pts_cam) < 15:
                        continue

                    # C. SOTA 3D Filtering: Downsample & Remove Outliers locally
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_cam)
                    
                    # Voxel downsample (0.05m = 5cm)
                    pcd = pcd.voxel_down_sample(voxel_size=0.05)
                    
                    # Statistical Outlier Removal to prevent OBB explosion
                    if len(pcd.points) > 20:
                        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    
                    clean_pts_cam = np.asarray(pcd.points, dtype=np.float32)
                    if len(clean_pts_cam) < 10:
                        continue

                    # D. Append to batch lists
                    object_names.append(det.class_name)
                    tracker_ids.append(str(det.instance_id))
                    embeddings_list.append(np.array(det.embedding, dtype=np.float32))
                    confidences.append(det.confidence)
                    points_cam_list.append(clean_pts_cam)

                # 3. Pass the batch to the existing Bipartite Matrix in mapper_v3.py
                if points_cam_list:
                    self.semantic_map.add_detections_batch(
                        object_names=object_names,
                        tracker_ids=tracker_ids,
                        detection_stamp=mask_msg.header.stamp,
                        camera_frame=self.camera_frame,
                        fixed_frame=self.map_frame,
                        embeddings_list=embeddings_list,
                        confidences=confidences,
                        points_cam_list=points_cam_list
                    )

                self.publish_semantic_map()
                if self.publish_stable_pointcloud_enabled:
                    self.publish_stable_pointcloud()
                    
        except Exception as ex:
            self.get_logger().error(f"[mapper_node_cpp_prep] sync error: {ex}")

    def publish_semantic_map(self):
        if not self.semantic_map.objects:
            return

        msg = SemanticObjectArray()
        append = msg.objects.append

        for map_id, entry in self.semantic_map.objects.items():
            obj = SemanticObject()
            obj.object_id = map_id
            obj.name = entry.current_name
            obj.frame = entry.frame
            obj.timestamp = entry.timestamp
            obj.pose_cam = Vector3(x=float(entry.pose_cam[0]), y=float(entry.pose_cam[1]), z=float(entry.pose_cam[2]))
            obj.pose_map = Vector3(x=float(entry.pose_map[0]), y=float(entry.pose_map[1]), z=float(entry.pose_map[2]))
            obj.occurrences = int(entry.occurrences)
            obj.similarity = float(entry.similarity)
            obj.confidence = float(entry.confidence_ema)

            if hasattr(obj, 'image_embedding'):
                obj.image_embedding = entry.image_embedding.tolist() if entry.image_embedding is not None else []

            append(obj)

        self.map_pub.publish(msg)

    def publish_stable_pointcloud(self):
        """Publishes all confirmed accumulated map points as object-colored PointCloud2 for RViz2."""
        if not self.semantic_map.objects:
            return

        cloud_points = []
        class_point_counts = {}
        total_points = 0
        for map_id, entry in self.semantic_map.objects.items():
            pts = entry.accumulated_points
            if pts is None or len(pts) == 0:
                continue
            class_name = entry.current_name
            color = self._class_to_color_rgb(str(map_id))
            rgb_packed = self._pack_rgb_as_float32(color[0], color[1], color[2])
            class_point_counts[class_name] = class_point_counts.get(class_name, 0) + len(pts)

            for p in pts:
                cloud_points.append((float(p[0]), float(p[1]), float(p[2]), rgb_packed))
            total_points += len(pts)

        if total_points == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, cloud_points)
        self.stable_cloud_pub.publish(cloud_msg)

        class_counts_log = ', '.join(
            f"{k}:{v}" for k, v in sorted(class_point_counts.items(), key=lambda kv: kv[0])
        )
        self.get_logger().info(
            "[mapper_node_v5:stable_cloud] "
            f"published_points={total_points} objects={len(self.semantic_map.objects)} "
            f"topic={self.stable_pointcloud_topic} classes={class_counts_log}",
            throttle_duration_sec=2.0,
        )

    def _class_to_color_rgb(self, class_name: str):
        """Deterministically maps class labels to bright RGB colors."""
        seed = zlib.crc32(class_name.encode('utf-8'))
        r = 60 + (seed & 0x7F)
        g = 60 + ((seed >> 8) & 0x7F)
        b = 60 + ((seed >> 16) & 0x7F)
        return (int(r), int(g), int(b))

    def _pack_rgb_as_float32(self, r: int, g: int, b: int) -> float:
        """Packs 8-bit RGB into PointCloud2 float32 rgb field format."""
        rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
        return struct.unpack('f', struct.pack('I', rgb_uint32))[0]

    def export_callback(self):
        with self.lock:
            try:
                # Periodic geometry cleanup: keep only largest cluster per object.
                # self.semantic_map.keep_only_largest_cluster()

                # NEW: Clean up duplicates before saving/publishing
                self.semantic_map.resolve_overlapping_duplicates()
                
                # Existing export logic
                self.semantic_map.export_to_json(self.output_dir, self.output_map_file)
                self.publish_semantic_map()
            except Exception as ex:
                self.get_logger().error(f"Export/Merge error: {ex}")

    def shutdown_callback(self):
        with self.lock:
            try:
                self.get_logger().info('[mapper_node_v5:shutdown] exporting final map')
                self.semantic_map.export_to_json(self.output_dir, 'map_v5_final.json')
                self.get_logger().info('[mapper_node_v5:shutdown] final export complete')
            except Exception as ex:
                self.get_logger().error(f"[mapper_node_v5] final export error: {ex}")
                self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    logger = rclpy.logging.get_logger('mapper_node_v5_main')
    logger.info('[mapper_node_v5:main] init complete, creating node')
    node = PointCloudMapperNodeV5()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        logger.info('[mapper_node_v5:main] spinning executor')
        executor.spin()
    except KeyboardInterrupt:
        logger.info('[mapper_node_v5:main] keyboard interrupt received')
        pass
    finally:
        logger.info('[mapper_node_v5:main] shutdown sequence start')
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()
        logger.info('[mapper_node_v5:main] shutdown complete')

if __name__ == '__main__':
    main()
