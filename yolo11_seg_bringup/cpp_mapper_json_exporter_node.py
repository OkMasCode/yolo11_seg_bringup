import json
import os
from typing import Dict, Any

import rclpy
from rclpy.node import Node

from yolo11_seg_interfaces.msg import SemanticObjectArray


class CppMapperJsonExporterNode(Node):
    def __init__(self) -> None:
        super().__init__("cpp_mapper_json_exporter_node")

        self.declare_parameter("input_topic", "/vision/semantic_map_v5")
        self.declare_parameter("output_dir", "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config")
        self.declare_parameter("output_map_file", "map_v6.json")
        self.declare_parameter("export_interval", 3.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.output_map_file = str(self.get_parameter("output_map_file").value)
        self.export_interval = float(self.get_parameter("export_interval").value)

        if self.export_interval <= 0.0:
            self.get_logger().warning(
                "[cpp_mapper_json_exporter] export_interval <= 0. Using default 3.0s"
            )
            self.export_interval = 3.0

        self.latest_map: Dict[str, Dict[str, Any]] = {}
        self.messages_seen = 0

        self.subscription = self.create_subscription(
            SemanticObjectArray,
            self.input_topic,
            self.semantic_map_callback,
            10,
        )

        self.timer = self.create_timer(self.export_interval, self.export_callback)

        self.get_logger().info(
            "[cpp_mapper_json_exporter:init] "
            f"input_topic={self.input_topic} output={os.path.join(self.output_dir, self.output_map_file)} "
            f"interval={self.export_interval:.2f}s"
        )

    def semantic_map_callback(self, msg: SemanticObjectArray) -> None:
        self.messages_seen += 1

        new_map: Dict[str, Dict[str, Any]] = {}
        for obj in msg.objects:
            map_id = obj.object_id if obj.object_id else "unknown"

            corners = [
                {
                    "x": float(pt.x),
                    "y": float(pt.y),
                    "z": float(pt.z),
                }
                for pt in obj.bbox_corners
            ]

            # Match legacy mapper_v5 export schema as closely as possible from available fields.
            new_map[map_id] = {
                "name": obj.name,
                "frame": obj.frame,
                "timestamp": {
                    "sec": int(obj.timestamp.sec),
                    "nanosec": int(obj.timestamp.nanosec),
                },
                "pose_map": {
                    "x": float(obj.pose_map.x),
                    "y": float(obj.pose_map.y),
                    "z": float(obj.pose_map.z),
                },
                "bbox_type": obj.bbox_type if obj.bbox_type else "unknown",
                "box_size": {
                    "x": float(obj.box_size.x),
                    "y": float(obj.box_size.y),
                    "z": float(obj.box_size.z),
                },
                "bbox_orientation": {
                    "x": float(obj.bbox_orientation.x),
                    "y": float(obj.bbox_orientation.y),
                    "z": float(obj.bbox_orientation.z),
                    "w": float(obj.bbox_orientation.w),
                },
                "bbox_corners": corners,
                "occurrences": int(obj.occurrences),
                "similarity": float(obj.similarity),
                "image_embedding": [float(v) for v in obj.image_embedding],
                "embedding_confidence": float(obj.confidence),
                "confidence": float(obj.confidence),
            }

        self.latest_map = new_map

        self.get_logger().info(
            "[cpp_mapper_json_exporter:rx] "
            f"msg={self.messages_seen} objects={len(msg.objects)}"
        )

    def export_callback(self) -> None:
        output_path = os.path.join(self.output_dir, self.output_map_file)

        if not self.latest_map:
            self.get_logger().warn(
                "[cpp_mapper_json_exporter:export] skip: no semantic map received yet",
                throttle_duration_sec=5.0,
            )
            return

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(self.latest_map, handle, indent=4)

            self.get_logger().info(
                "[cpp_mapper_json_exporter:export] "
                f"wrote {len(self.latest_map)} objects to {output_path}"
            )
        except Exception as exc:
            self.get_logger().error(
                "[cpp_mapper_json_exporter:export] "
                f"failed to write json: {exc}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CppMapperJsonExporterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
