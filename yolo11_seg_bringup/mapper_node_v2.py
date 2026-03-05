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
    - Exports a richer JSON payload with runtime stats for comparison.
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

        self.dm_topic = self.get_parameter('detection_message').value
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_dir').value
        self.export_interval = float(self.get_parameter('export_interval').value)
        self.load_map_on_start = self.get_parameter('load_map_on_start').value
        self.input_map_file = self.get_parameter('input_map_file').value
        self.output_map_file = self.get_parameter('output_map_file').value

        # ---------- ROS interfaces ---------- #
        self.semantic_map_topic = '/vision/semantic_map_v2'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.semantic_map = SemanticObjectMapV2(self.tf_buffer, self)

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
