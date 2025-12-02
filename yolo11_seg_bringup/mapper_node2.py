#!/usr/bin/env python3

import numpy as np
import os
from collections import defaultdict
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Vector3, Point
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from ultralytics import YOLO

from yolo11_seg_bringup.mapper2 import SemanticObjectMap
from yolo11_seg_interfaces.msg import DetectedObject

class PointCloudMapperNode(Node):
    def __init__(self):
        super().__init__("pointcloud_mapper_node")

        # ============= Parameters ============= #

        self.declare_parameter("custom_message", "/yolo/custom_message") # Added
        self.declare_parameter("output_dir", "/home/sensor/ros2_ws/src/yolo11_seg_bringup")
        self.declare_parameter("export_interval", 5.0)
        self.declare_parameter("map_frame", "camera_color_optical_frame")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        
        self.cm_topic = self.get_parameter("custom_message").value # Added
        self.output_dir = self.get_parameter("output_dir").value
        self.export_interval = float(self.get_parameter("export_interval").value)
        self.map_frame = self.get_parameter("map_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        # ============= Initialization ============= #

        qos_sensor = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.semantic_map = SemanticObjectMap(self.tf_buffer, self)

        self.cm_sub = self.create_subscription(DetectedObject, self.cm_topic, self.custom_callback, qos_profile=qos_sensor) # Added

        self.export_timer = self.create_timer(self.export_interval, self.export_callback)

        self.lock = threading.Lock()
        
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
                    distance_threshold=0.2
                )

                self.get_logger().debug(
                    f"Detected {class_name} (inst {instance_id}) at "
                    f"({centroid.x:.3f}, {centroid.y:.3f}, {centroid.z:.3f})"
                )

        except Exception as e:
            self.get_logger().error(f"Error processing DetectedObject: {e}")

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
            except Exception as e:
                self.get_logger().error(f"Export failed: {e}")

        self.publish_centroids()

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