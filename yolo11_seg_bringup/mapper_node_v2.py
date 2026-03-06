import threading

import rclpy
import tf2_ros
from geometry_msgs.msg import Vector3
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from yolo11_seg_bringup.mapper_v2 import SemanticObjectMapV2
from yolo11_seg_interfaces.msg import DetectedObject, SemanticObject, SemanticObjectArray


class PointCloudMapperNodeV2(Node):
    """
    V2 mapper node for benchmarking against the baseline mapper node.

    Main difference from the baseline:
    - Uses SemanticObjectMapV2 with explicit tracker_id -> map_id bindings.
    - Exports baseline-compatible JSON so downstream consumers can be shared.
    """

    def __init__(self):
        super().__init__('pointcloud_mapper_node_v2')

        # ---------- Parameters ---------- #
        self.declare_parameter('detection_message', '/vision/detections')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('output_dir', '/workspaces/ros2_ws/src/yolo11_seg_bringup/config/')
        self.declare_parameter('export_interval', 5.0)
        self.declare_parameter('load_map_on_start', False)
        self.declare_parameter('input_map_file', 'map_v2.json')
        self.declare_parameter('output_map_file', 'map_v2.json')
        self.declare_parameter('use_bbox_center_pose', False)
        self.declare_parameter('binding_ttl_sec', 4.0)
        self.declare_parameter('confirmation_min_age_sec', 0.6)
        self.declare_parameter('min_input_confidence', 0.45)
        self.declare_parameter('min_avg_confidence_for_promotion', 0.60)
        self.declare_parameter('min_class_vote_ratio_for_promotion', 0.70)
        self.declare_parameter('min_detection_depth_m', 0.20)
        self.declare_parameter('max_detection_depth_m', 8.00)
        self.declare_parameter('min_box_extent_m', 0.03)
        self.declare_parameter('max_box_extent_m', 4.00)
        self.declare_parameter('min_box_volume_m3', 0.0003)
        self.declare_parameter('max_box_volume_m3', 25.0)
        self.declare_parameter('min_same_class_gate', 0.55)
        self.declare_parameter('min_size_similarity_for_overlap', 0.25)
        self.declare_parameter('min_association_iou', 0.10)
        self.declare_parameter('merge_min_iou', 0.18)
        self.declare_parameter('merge_min_size_similarity', 0.4)
        self.declare_parameter('dedup_only_same_class', True)
        self.declare_parameter('enable_relaxed_large_object_dedup', True)
        self.declare_parameter('large_object_min_extent', 0.90)
        self.declare_parameter('relaxed_large_dedup_distance', 1.00)
        self.declare_parameter('relaxed_large_dedup_min_size_similarity', 0.30)
        self.declare_parameter('recovery_ttl_sec', 6.0)
        self.declare_parameter('max_recovery_distance', 0.45)
        self.declare_parameter('min_recovery_iou', 0.20)
        self.declare_parameter('min_recovery_size_similarity', 0.35)
        self.declare_parameter('enable_strict_active_rebind', True)
        self.declare_parameter('min_occurrences_for_active_rebind', 4)
        self.declare_parameter('max_active_rebind_distance', 0.75)
        self.declare_parameter('min_active_rebind_iou', 0.08)
        self.declare_parameter('min_active_rebind_size_similarity', 0.25)
        self.declare_parameter('active_rebind_ambiguity_margin', 0.04)
        self.declare_parameter('enable_online_dedup', True)
        self.declare_parameter('enable_same_class_containment_merge', True)
        self.declare_parameter('containment_margin', 0.1)
        self.declare_parameter('min_containment_size_similarity', 0.1)
        self.declare_parameter('enable_detection_bbox_merge', True)
        self.declare_parameter('detection_bbox_merge_scale', 1.15)
        self.declare_parameter('detection_bbox_merge_margin', 0.05)
        self.declare_parameter('detection_bbox_merge_min_extent', 0.90)
        self.declare_parameter('detection_bbox_merge_min_volume', 0.20)
        self.declare_parameter('small_object_max_extent', 0.80)
        self.declare_parameter('small_object_assoc_min_iou', 0.22)
        self.declare_parameter('small_object_assoc_min_size_similarity', 0.35)
        self.declare_parameter('small_object_merge_max_distance', 0.40)
        self.declare_parameter('small_object_merge_min_iou', 0.30)
        self.declare_parameter('small_object_merge_min_size_similarity', 0.55)
        self.declare_parameter('max_pose_step_unlocked_base', 0.1)
        self.declare_parameter('max_pose_step_unlocked_gain', 0.04)

        self.dm_topic = self.get_parameter('detection_message').value
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_dir').value
        self.export_interval = float(self.get_parameter('export_interval').value)
        self.load_map_on_start = self.get_parameter('load_map_on_start').value
        self.input_map_file = self.get_parameter('input_map_file').value
        self.output_map_file = self.get_parameter('output_map_file').value
        self.use_bbox_center_pose = bool(self.get_parameter('use_bbox_center_pose').value)
        self.binding_ttl_sec = float(self.get_parameter('binding_ttl_sec').value)
        self.confirmation_min_age_sec = float(self.get_parameter('confirmation_min_age_sec').value)
        self.min_input_confidence = float(self.get_parameter('min_input_confidence').value)
        self.min_avg_confidence_for_promotion = float(self.get_parameter('min_avg_confidence_for_promotion').value)
        self.min_class_vote_ratio_for_promotion = float(self.get_parameter('min_class_vote_ratio_for_promotion').value)
        self.min_detection_depth_m = float(self.get_parameter('min_detection_depth_m').value)
        self.max_detection_depth_m = float(self.get_parameter('max_detection_depth_m').value)
        self.min_box_extent_m = float(self.get_parameter('min_box_extent_m').value)
        self.max_box_extent_m = float(self.get_parameter('max_box_extent_m').value)
        self.min_box_volume_m3 = float(self.get_parameter('min_box_volume_m3').value)
        self.max_box_volume_m3 = float(self.get_parameter('max_box_volume_m3').value)
        self.min_same_class_gate = float(self.get_parameter('min_same_class_gate').value)
        self.min_size_similarity_for_overlap = float(self.get_parameter('min_size_similarity_for_overlap').value)
        self.min_association_iou = float(self.get_parameter('min_association_iou').value)
        self.merge_min_iou = float(self.get_parameter('merge_min_iou').value)
        self.merge_min_size_similarity = float(self.get_parameter('merge_min_size_similarity').value)
        self.dedup_only_same_class = bool(self.get_parameter('dedup_only_same_class').value)
        self.enable_relaxed_large_object_dedup = bool(self.get_parameter('enable_relaxed_large_object_dedup').value)
        self.large_object_min_extent = float(self.get_parameter('large_object_min_extent').value)
        self.relaxed_large_dedup_distance = float(self.get_parameter('relaxed_large_dedup_distance').value)
        self.relaxed_large_dedup_min_size_similarity = float(self.get_parameter('relaxed_large_dedup_min_size_similarity').value)
        self.recovery_ttl_sec = float(self.get_parameter('recovery_ttl_sec').value)
        self.max_recovery_distance = float(self.get_parameter('max_recovery_distance').value)
        self.min_recovery_iou = float(self.get_parameter('min_recovery_iou').value)
        self.min_recovery_size_similarity = float(self.get_parameter('min_recovery_size_similarity').value)
        self.enable_strict_active_rebind = bool(self.get_parameter('enable_strict_active_rebind').value)
        self.min_occurrences_for_active_rebind = int(self.get_parameter('min_occurrences_for_active_rebind').value)
        self.max_active_rebind_distance = float(self.get_parameter('max_active_rebind_distance').value)
        self.min_active_rebind_iou = float(self.get_parameter('min_active_rebind_iou').value)
        self.min_active_rebind_size_similarity = float(self.get_parameter('min_active_rebind_size_similarity').value)
        self.active_rebind_ambiguity_margin = float(self.get_parameter('active_rebind_ambiguity_margin').value)
        self.enable_online_dedup = bool(self.get_parameter('enable_online_dedup').value)
        self.enable_same_class_containment_merge = bool(self.get_parameter('enable_same_class_containment_merge').value)
        self.containment_margin = float(self.get_parameter('containment_margin').value)
        self.min_containment_size_similarity = float(self.get_parameter('min_containment_size_similarity').value)
        self.enable_detection_bbox_merge = bool(self.get_parameter('enable_detection_bbox_merge').value)
        self.detection_bbox_merge_scale = float(self.get_parameter('detection_bbox_merge_scale').value)
        self.detection_bbox_merge_margin = float(self.get_parameter('detection_bbox_merge_margin').value)
        self.detection_bbox_merge_min_extent = float(self.get_parameter('detection_bbox_merge_min_extent').value)
        self.detection_bbox_merge_min_volume = float(self.get_parameter('detection_bbox_merge_min_volume').value)
        self.small_object_max_extent = float(self.get_parameter('small_object_max_extent').value)
        self.small_object_assoc_min_iou = float(self.get_parameter('small_object_assoc_min_iou').value)
        self.small_object_assoc_min_size_similarity = float(self.get_parameter('small_object_assoc_min_size_similarity').value)
        self.small_object_merge_max_distance = float(self.get_parameter('small_object_merge_max_distance').value)
        self.small_object_merge_min_iou = float(self.get_parameter('small_object_merge_min_iou').value)
        self.small_object_merge_min_size_similarity = float(self.get_parameter('small_object_merge_min_size_similarity').value)
        self.max_pose_step_unlocked_base = float(self.get_parameter('max_pose_step_unlocked_base').value)
        self.max_pose_step_unlocked_gain = float(self.get_parameter('max_pose_step_unlocked_gain').value)

        # ---------- ROS interfaces ---------- #
        self.semantic_map_topic = '/vision/semantic_map_v2'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.semantic_map = SemanticObjectMapV2(self.tf_buffer, self)
        self.semantic_map.use_bbox_center_pose = self.use_bbox_center_pose
        self.semantic_map.binding_ttl_sec = self.binding_ttl_sec
        self.semantic_map.confirmation_min_age_sec = self.confirmation_min_age_sec
        self.semantic_map.min_input_confidence = self.min_input_confidence
        self.semantic_map.min_avg_confidence_for_promotion = self.min_avg_confidence_for_promotion
        self.semantic_map.min_class_vote_ratio_for_promotion = self.min_class_vote_ratio_for_promotion
        self.semantic_map.min_detection_depth_m = self.min_detection_depth_m
        self.semantic_map.max_detection_depth_m = self.max_detection_depth_m
        self.semantic_map.min_box_extent_m = self.min_box_extent_m
        self.semantic_map.max_box_extent_m = self.max_box_extent_m
        self.semantic_map.min_box_volume_m3 = self.min_box_volume_m3
        self.semantic_map.max_box_volume_m3 = self.max_box_volume_m3
        self.semantic_map.min_same_class_gate = self.min_same_class_gate
        self.semantic_map.min_size_similarity_for_overlap = self.min_size_similarity_for_overlap
        self.semantic_map.min_association_iou = self.min_association_iou
        self.semantic_map.merge_min_iou = self.merge_min_iou
        self.semantic_map.merge_min_size_similarity = self.merge_min_size_similarity
        self.semantic_map.dedup_only_same_class = self.dedup_only_same_class
        self.semantic_map.enable_relaxed_large_object_dedup = self.enable_relaxed_large_object_dedup
        self.semantic_map.large_object_min_extent = self.large_object_min_extent
        self.semantic_map.relaxed_large_dedup_distance = self.relaxed_large_dedup_distance
        self.semantic_map.relaxed_large_dedup_min_size_similarity = self.relaxed_large_dedup_min_size_similarity
        self.semantic_map.recovery_ttl_sec = self.recovery_ttl_sec
        self.semantic_map.max_recovery_distance = self.max_recovery_distance
        self.semantic_map.min_recovery_iou = self.min_recovery_iou
        self.semantic_map.min_recovery_size_similarity = self.min_recovery_size_similarity
        self.semantic_map.enable_strict_active_rebind = self.enable_strict_active_rebind
        self.semantic_map.min_occurrences_for_active_rebind = self.min_occurrences_for_active_rebind
        self.semantic_map.max_active_rebind_distance = self.max_active_rebind_distance
        self.semantic_map.min_active_rebind_iou = self.min_active_rebind_iou
        self.semantic_map.min_active_rebind_size_similarity = self.min_active_rebind_size_similarity
        self.semantic_map.active_rebind_ambiguity_margin = self.active_rebind_ambiguity_margin
        self.semantic_map.enable_online_dedup = self.enable_online_dedup
        self.semantic_map.enable_same_class_containment_merge = self.enable_same_class_containment_merge
        self.semantic_map.containment_margin = self.containment_margin
        self.semantic_map.min_containment_size_similarity = self.min_containment_size_similarity
        self.semantic_map.enable_detection_bbox_merge = self.enable_detection_bbox_merge
        self.semantic_map.detection_bbox_merge_scale = self.detection_bbox_merge_scale
        self.semantic_map.detection_bbox_merge_margin = self.detection_bbox_merge_margin
        self.semantic_map.detection_bbox_merge_min_extent = self.detection_bbox_merge_min_extent
        self.semantic_map.detection_bbox_merge_min_volume = self.detection_bbox_merge_min_volume
        self.semantic_map.small_object_max_extent = self.small_object_max_extent
        self.semantic_map.small_object_assoc_min_iou = self.small_object_assoc_min_iou
        self.semantic_map.small_object_assoc_min_size_similarity = self.small_object_assoc_min_size_similarity
        self.semantic_map.small_object_merge_max_distance = self.small_object_merge_max_distance
        self.semantic_map.small_object_merge_min_iou = self.small_object_merge_min_iou
        self.semantic_map.small_object_merge_min_size_similarity = self.small_object_merge_min_size_similarity
        self.semantic_map.max_pose_step_unlocked_base = self.max_pose_step_unlocked_base
        self.semantic_map.max_pose_step_unlocked_gain = self.max_pose_step_unlocked_gain

        if self.load_map_on_start:
            self.semantic_map.load_from_json(self.output_dir, self.input_map_file)
            self.get_logger().info(
                f"[mapper_node_v2] loaded map from {self.input_map_file} with "
                f"{len(self.semantic_map.objects)} objects"
            )

        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.dm_sub = self.create_subscription(
            DetectedObject,
            self.dm_topic,
            self.detection_callback,
            qos_profile=qos_sensor,
        )
        self.map_pub = self.create_publisher(SemanticObjectArray, self.semantic_map_topic, 10)
        self.export_timer = self.create_timer(self.export_interval, self.export_callback)

        self.lock = threading.Lock()

        self.get_logger().info(
            "[mapper_node_v2] initialized. "
            f"input={self.dm_topic}, output_topic={self.semantic_map_topic}, output_dir={self.output_dir}"
        )

    def detection_callback(self, msg: DetectedObject):
        """Process one detection, update map, and publish the current map snapshot."""
        try:
            with self.lock:
                class_name = msg.object_name
                tracker_id = str(msg.object_id)

                self.semantic_map.add_detection(
                    object_name=class_name,
                    tracker_id=tracker_id,
                    pose_in_camera=msg.centroid,
                    detection_stamp=msg.timestamp,
                    camera_frame=self.camera_frame,
                    fixed_frame=self.map_frame,
                    embeddings=msg.embedding,
                    similarity=msg.similarity,
                    confidence=msg.confidence,
                    box_min=(msg.box_min.x, msg.box_min.y, msg.box_min.z),
                    box_max=(msg.box_max.x, msg.box_max.y, msg.box_max.z),
                )

                self.publish_semantic_map()

        except Exception as ex:
            self.get_logger().error(f"[mapper_node_v2] Error processing detection: {ex}")

    def publish_semantic_map(self):
        """
        Publish current persistent map in the same message type as baseline.

        The IDs now represent persistent map object IDs (map_obj_XXXXXX), not per-frame detection IDs.
        """
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
            obj.pose_cam = Vector3(
                x=float(entry.pose_cam[0]),
                y=float(entry.pose_cam[1]),
                z=float(entry.pose_cam[2]),
            )
            obj.pose_map = Vector3(
                x=float(entry.pose_map[0]),
                y=float(entry.pose_map[1]),
                z=float(entry.pose_map[2]),
            )
            obj.occurrences = int(entry.occurrences)
            obj.similarity = float(entry.similarity)
            obj.confidence = float(entry.confidence_ema)

            if hasattr(obj, 'image_embedding'):
                obj.image_embedding = entry.image_embedding.tolist() if entry.image_embedding is not None else []

            append(obj)

        self.map_pub.publish(msg)

    def export_callback(self):
        """Periodic map export used for offline benchmarking and persistence."""
        with self.lock:
            try:
                self.semantic_map.export_to_json(self.output_dir, self.output_map_file)
            except Exception as ex:
                self.get_logger().error(f"[mapper_node_v2] export error: {ex}")

    def shutdown_callback(self):
        """Final export with a dedicated file name to preserve last state."""
        with self.lock:
            try:
                self.semantic_map.export_to_json(self.output_dir, 'map_v2_final.json')
            except Exception as ex:
                self.get_logger().error(f"[mapper_node_v2] final export failed: {ex}")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudMapperNodeV2()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
