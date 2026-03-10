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
    """Minimal semantic mapper node with clear and tunable behavior."""

    def __init__(self):
        super().__init__('pointcloud_mapper_node_v2')

        # Core I/O.
        self.declare_parameter('detection_message', '/vision/detections')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('output_dir', '/workspaces/ros2_ws/src/yolo11_seg_bringup/config/')
        self.declare_parameter('export_interval', 5.0)
        self.declare_parameter('load_map_on_start', False)
        self.declare_parameter('input_map_file', 'map_v3.json')
        self.declare_parameter('output_map_file', 'map_v3.json')

        # False-positive suppression.
        self.declare_parameter('min_input_confidence', 0.55)
        self.declare_parameter('confirmation_min_hits', 10)
        self.declare_parameter('confirmation_min_age_sec', 2)
        self.declare_parameter('min_avg_confidence_for_promotion', 0.50)

        # Geometry gating.
        self.declare_parameter('min_detection_depth_m', 0.25)
        self.declare_parameter('max_detection_depth_m', 4.0)
        self.declare_parameter('min_association_iou', 0.12)
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

        self.dm_topic = self.get_parameter('detection_message').value
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_dir').value
        self.export_interval = float(self.get_parameter('export_interval').value)
        self.load_map_on_start = bool(self.get_parameter('load_map_on_start').value)
        self.input_map_file = self.get_parameter('input_map_file').value
        self.output_map_file = self.get_parameter('output_map_file').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.semantic_map = SemanticObjectMapV2(self.tf_buffer, self)

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

        if self.load_map_on_start:
            self.semantic_map.load_from_json(self.output_dir, self.input_map_file)

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
        self.map_pub = self.create_publisher(SemanticObjectArray, '/vision/semantic_map_v3', 10)
        self.export_timer = self.create_timer(self.export_interval, self.export_callback)

        self.lock = threading.Lock()

        self.get_logger().info(
            f"[mapper_node_v2] ready. input={self.dm_topic}, output_dir={self.output_dir}, export={self.output_map_file}"
        )

    def detection_callback(self, msg: DetectedObject):
        try:
            with self.lock:
                pre_confirmed = len(self.semantic_map.objects)
                pre_tentative = len(self.semantic_map.tentative_tracks)

                created = self.semantic_map.add_detection(
                    object_name=msg.object_name,
                    tracker_id=str(msg.object_id),
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

                post_confirmed = len(self.semantic_map.objects)
                post_tentative = len(self.semantic_map.tentative_tracks)
                bound_map_id = self.semantic_map.track_to_map.get(str(msg.object_id), "-")

                if created or post_confirmed > pre_confirmed:
                    event = "new_confirmed"
                elif post_confirmed < pre_confirmed:
                    event = "merged_confirmed"
                elif post_tentative > pre_tentative:
                    event = "new_tentative"
                elif post_tentative < pre_tentative:
                    event = "tentative_transition"
                else:
                    event = "updated"

                self.get_logger().info(
                    "[det_state] "
                    f"event={event} name={msg.object_name} track={msg.object_id} "
                    f"conf={float(msg.confidence):.3f} sim={float(msg.similarity):.2f} "
                    f"emb={'yes' if len(msg.embedding) > 0 else 'no'} "
                    f"confirmed={post_confirmed} tentative={post_tentative} bound={bound_map_id}"
                )

                self.publish_semantic_map()
        except Exception as ex:
            self.get_logger().error(f"[mapper_node_v2] detection error: {ex}")

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

    def export_callback(self):
        with self.lock:
            try:
                self.semantic_map.export_to_json(self.output_dir, self.output_map_file)
            except Exception as ex:
                self.get_logger().error(f"[mapper_node_v2] export error: {ex}")

    def shutdown_callback(self):
        with self.lock:
            try:
                self.semantic_map.export_to_json(self.output_dir, 'map_v3_final.json')
            except Exception as ex:
                self.get_logger().error(f"[mapper_node_v2] final export error: {ex}")


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
