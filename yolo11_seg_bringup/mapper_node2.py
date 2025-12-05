#!/usr/bin/env python3

import numpy as np
import os
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Vector3
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from yolo11_seg_bringup.mapper2 import SemanticObjectMap
from yolo11_seg_interfaces.msg import DetectedObject, SemanticObject, SemanticObjectArray

class PointCloudMapperNode(Node):
    def __init__(self):
        super().__init__("pointcloud_mapper_node")

        # ============= Parameters ============= #

        self.declare_parameter("detection_message", "/yolo/detections")
        self.declare_parameter("output_dir", "/home/sensor/ros2_ws/src/yolo11_seg_bringup")
        self.declare_parameter("export_interval", 5.0)
        self.declare_parameter("map_frame", "camera_color_optical_frame")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("semantic_map_topic", "/yolo/semantic_map")
        
        self.cm_topic = self.get_parameter("detection_message").value
        self.output_dir = self.get_parameter("output_dir").value
        self.export_interval = float(self.get_parameter("export_interval").value)
        self.map_frame = self.get_parameter("map_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.semantic_map_topic = self.get_parameter("semantic_map_topic").value

        # ============= Initialization ============= #

        qos_sensor = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.semantic_map = SemanticObjectMap(self.tf_buffer, self)

        self.cm_sub = self.create_subscription(DetectedObject, self.cm_topic, self.custom_callback, qos_profile=qos_sensor)
        self.map_pub = self.create_publisher(SemanticObjectArray, self.semantic_map_topic, 10)
        
        self.export_timer = self.create_timer(self.export_interval, self.export_callback)

        self.lock = threading.Lock()
        
        # ============= Rate Tracking ============= #
        self.detection_count = 0
        self.detection_start_time = self.get_clock().now()
        self.export_count = 0
        self.export_start_time = self.get_clock().now()
        
        self.get_logger().info(f"PointCloud Mapper Node initialized. " f"Subscribing to {self.cm_topic}, output to {self.output_dir}")

    def custom_callback(self, msg: DetectedObject):
        """
        Process incoming DetectedObject message data and update the semantic map.
        """
        try:
            with self.lock:
                # Extract fields from DetectedObject message
                class_name = msg.object_name
                instance_id = msg.object_id
                centroid = msg.centroid
                timestamp = msg.timestamp
                embedding = msg.embedding
                goal_embedding = msg.text_embedding

                # Create unique object ID
                object_id = (
                    f"{class_name}_inst{instance_id}_"
                    f"{timestamp.sec}_{timestamp.nanosec}"
                )

                # All checks to avoid double detections is inside the add_detection method
                self.semantic_map.add_detection(
                    object_name=class_name,
                    object_id=object_id,
                    pose_in_camera=centroid,
                    detection_stamp=timestamp,
                    camera_frame=self.camera_frame,
                    fixed_frame=self.map_frame,
                    distance_threshold=0.2,
                    embeddings=embedding,
                    goal_embedding=goal_embedding
                )

                self.publish_semantic_map_locked()

                self.get_logger().info(
                    f"Detected {class_name} (inst {instance_id}) at "
                    f"({centroid.x:.3f}, {centroid.y:.3f}, {centroid.z:.3f})"
                )
                
                # Update detection rate tracking
                self.detection_count += 1
                elapsed = (self.get_clock().now() - self.detection_start_time).nanoseconds / 1e9
                if elapsed >= 1.0:  # Log rate every second
                    rate = self.detection_count / elapsed
                    self.get_logger().info(f"Detection processing rate: {rate:.2f} Hz")
                    self.detection_count = 0
                    self.detection_start_time = self.get_clock().now()

        except Exception as e:
            self.get_logger().error(f"Error processing DetectedObject: {e}")

    def publish_semantic_map_locked(self):
        """Publish all stored objects as SemanticObjectArray. Caller must hold self.lock."""
        if not self.semantic_map.objects:
            return

        msg = SemanticObjectArray()
        # Preallocate list for speed
        msg.objects = []
        append_obj = msg.objects.append

        for object_id, entry in self.semantic_map.objects.items():
            obj = SemanticObject()
            obj.object_id = object_id
            obj.name = entry.name
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
            obj.similarity = float(entry.similarity) if entry.similarity is not None else 0.0
            append_obj(obj)

        self.map_pub.publish(msg)

    def export_callback(self):
        """
        Export the current semantic map stored to a CSV file.
        """
        with self.lock:
            try:
                self.semantic_map.export_to_csv(
                    directory_path=self.output_dir,
                    file="detections.csv"
                )
                self.get_logger().info(
                    f"Exported {len(self.semantic_map.objects)} detections to {self.output_dir}/detections.csv"
                )
                self.semantic_map.export_to_json(
                    directory_path="/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/",
                    file="map.json"
                )
                # Update export rate tracking
                self.export_count += 1
                elapsed = (self.get_clock().now() - self.export_start_time).nanoseconds / 1e9
                if elapsed > 0:
                    rate = self.export_count / elapsed
                    self.get_logger().info(f"Export rate: {rate:.2f} Hz (1/{self.export_interval:.1f}s expected)")
                    
            except Exception as e:
                self.get_logger().error(f"Export failed: {e}")

    def shutdown_callback(self):
        """
        Final export of the semantic map to a CSV file during node shutdown.
        """
        with self.lock:
            try:
                self.semantic_map.export_to_csv(
                    directory_path=self.output_dir,
                    file="detections_final.csv"
                )
                self.get_logger().info(
                    f"Final export: {len(self.semantic_map.objects)} detections to {self.output_dir}/detections_final.csv"
                )
            except Exception as e:
                self.get_logger().error(f"Final export failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudMapperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()