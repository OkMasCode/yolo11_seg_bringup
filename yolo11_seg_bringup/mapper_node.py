#!/usr/bin/env python3
"""
PointCloud Mapper Node: Real-time object tracking from YOLO11 3D segmentation

Subscribe:
  - /yolo/pointcloud (sensor_msgs/PointCloud2) - 3D points with class_id field

Publishes:
  - None

Functionality:
  - Extracts class_id from pointcloud field
  - Computes centroid of each object cluster
  - Uses SemanticObjectMap to track objects in real-time
  - Exports detections to CSV periodically and on shutdown
"""

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

# COCO80 class names (index = class_id)
CLASS_NAMES = [
    "person",         # 0
    "bicycle",        # 1
    "car",            # 2
    "motorcycle",     # 3
    "airplane",       # 4
    "bus",            # 5
    "train",          # 6
    "truck",          # 7
    "boat",           # 8
    "traffic light",  # 9
    "fire hydrant",   # 10
    "stop sign",      # 11
    "parking meter",  # 12
    "bench",          # 13
    "bird",           # 14
    "cat",            # 15
    "dog",            # 16
    "horse",          # 17
    "sheep",          # 18
    "cow",            # 19
    "elephant",       # 20
    "bear",           # 21
    "zebra",          # 22
    "giraffe",        # 23
    "backpack",       # 24
    "umbrella",       # 25
    "handbag",        # 26
    "tie",            # 27
    "suitcase",       # 28
    "frisbee",        # 29
    "skis",           # 30
    "snowboard",      # 31
    "sports ball",    # 32
    "kite",           # 33
    "baseball bat",   # 34
    "baseball glove", # 35
    "skateboard",     # 36
    "surfboard",      # 37
    "tennis racket",  # 38
    "bottle",         # 39
    "wine glass",     # 40
    "cup",            # 41
    "fork",           # 42
    "knife",          # 43
    "spoon",          # 44
    "bowl",           # 45
    "banana",         # 46
    "apple",          # 47
    "sandwich",       # 48
    "orange",         # 49
    "broccoli",       # 50
    "carrot",         # 51
    "hot dog",        # 52
    "pizza",          # 53
    "donut",          # 54
    "cake",           # 55
    "chair",          # 56
    "couch",          # 57
    "potted plant",   # 58
    "bed",            # 59
    "dining table",   # 60
    "toilet",         # 61
    "tv",             # 62
    "laptop",         # 63
    "mouse",          # 64
    "remote",         # 65
    "keyboard",       # 66
    "cell phone",     # 67
    "microwave",      # 68
    "oven",           # 69
    "toaster",        # 70
    "sink",           # 71
    "refrigerator",   # 72
    "book",           # 73
    "clock",          # 74
    "vase",           # 75
    "scissors",       # 76
    "teddy bear",     # 77
    "hair drier",     # 78
    "toothbrush",     # 79
]

def class_id_to_name(class_id: int) -> str:
    """Safe lookup with fallback for out-of-range IDs."""
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


class PointCloudMapperNode(Node):
    def __init__(self):
        super().__init__("pointcloud_mapper_node")

        # Parameters
        self.declare_parameter("pointcloud_topic", "/yolo/pointcloud") # name of pointcloud topic
        self.declare_parameter("output_dir", "/home/sensor/ros2_ws/src/yolo11_seg_bringup") # Directory to save CSV exports
        self.declare_parameter("export_interval", 10.0)  # seconds 
        self.declare_parameter("map_frame", "camera_color_optical_frame")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("model_path", "/home/sensor/yolo11n-seg.engine") # I need to use the vocabulary of the model to get the class names (must find another way)

        # Get parameters
        self.pc_topic = self.get_parameter("pointcloud_topic").value
        self.output_dir = self.get_parameter("output_dir").value
        self.export_interval = float(self.get_parameter("export_interval").value)
        self.map_frame = self.get_parameter("map_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        model_path = self.get_parameter("model_path").value

        # QoS profile
        # necessary for synced sensor data
        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # SemanticObjectMap
        self.semantic_map = SemanticObjectMap(self.tf_buffer, self)

        # Subscribe
        self.pc_sub = self.create_subscription(
            PointCloud2, self.pc_topic, self.pointcloud_callback, qos_profile=qos_sensor
        )

        # Publisher for centroid markers
        self.marker_pub = self.create_publisher(MarkerArray, "/yolo/centroids", 10)

        # Export timer
        self.export_timer = self.create_timer(self.export_interval, self.export_callback)

        # Lock for thread safety
        self.lock = threading.Lock()

        # Marker ID counter
        self.marker_id_counter = 0

        self.get_logger().info(
            f"PointCloud Mapper Node initialized. "
            f"Subscribing to {self.pc_topic}, output to {self.output_dir}"
        )

    def pointcloud_callback(self, msg: PointCloud2):
        """
        Process incoming pointcloud, extract class_id and centroid,
        and add to SemanticObjectMap.
        """
        try:
            # Convert pointcloud to list of points
            points_gen = point_cloud2.read_points(msg, skip_nans=True)
            points_list = list(points_gen)

            if not points_list:
                return

            # Group points by class_id
            class_points = defaultdict(list)
            cluster_points = defaultdict(list)
            for point in points_list:
                x, y, z = point[0], point[1], point[2]
                
                # Extract class_id from the pointcloud field (now at index 4)
                class_id = int(point[4]) if len(point) > 4 else -1

                # Extract instance_id from the pointcloud field (now at index 5)
                instance_id = int(point[5]) if len(point) > 5 else 0
                
                cluster_points[(class_id, instance_id)].append((x, y, z))
                # class_points[class_id].append((x, y, z))

                
            # For each class, compute centroid and add to map
            with self.lock:
                for (class_id, instance_id), points in cluster_points.items():
                    class_name = class_id_to_name(class_id)

                    points_array = np.array(points)
                    centroid = points_array.mean(axis=0)

                    # Create Vector3 for centroid
                    pose_vector = Vector3(
                        x=float(centroid[0]),
                        y=float(centroid[1]),
                        z=float(centroid[2]),
                    )

                    # Create timestamp
                    timestamp = Time(
                        sec=msg.header.stamp.sec,
                        nanosec=msg.header.stamp.nanosec
                    )

                    # Create unique object_id
                    # object_id = f"{class_name}_{msg.header.stamp.sec}_{msg.header.stamp.nanosec}_{class_id}"
                    object_id = (
                        f"{class_name}_inst{instance_id}_"
                        f"{msg.header.stamp.sec}_{msg.header.stamp.nanosec}"
                    )
                    # Add to semantic map using mapper2 API
                    self.semantic_map.add_detection(
                        object_name=class_name,
                        object_id=object_id,
                        pose_in_camera=pose_vector,
                        detection_stamp=timestamp,
                        camera_frame=msg.header.frame_id or self.camera_frame,
                        fixed_frame=self.map_frame,
                        distance_threshold=0.2
                    )

                    # self.get_logger().debug(
                    #     f"Detected {class_name} at ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})"
                    # )
                    self.get_logger().debug(
                        f"Detected {class_name} (inst {instance_id}) at "
                        f"({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})"
                    )

        except Exception as e:
            self.get_logger().error(f"Error processing pointcloud: {e}")

    def create_centroid_marker(self, class_name: str, centroid: Vector3, class_id: int, marker_id: int):
        """Create a visualization sphere marker for a detected object centroid.

        Note: `centroid` should be a `geometry_msgs.msg.Vector3` or any object
        with `.x/.y/.z` attributes. We set the `pose.position` components
        individually because `pose.position` expects a `Point` message, not a
        `Vector3`.
        """
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        # assign position components individually
        marker.pose.position.x = float(centroid.x)
        marker.pose.position.y = float(centroid.y)
        marker.pose.position.z = float(centroid.z)
        marker.pose.orientation.w = 1.0

        # Size of sphere (10cm diameter)
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Color based on class_name (deterministic colors)
        h = abs(hash(class_name))
        r = ((h >> 0) & 0xFF) / 255.0
        g = ((h >> 8) & 0xFF) / 255.0
        b = ((h >> 16) & 0xFF) / 255.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 0.8

        return marker

    def publish_centroids(self):
        """Publish centroid markers from the semantic map."""
        marker_array = MarkerArray()
        
        with self.lock:
            for idx, (obj_id, entry) in enumerate(self.semantic_map.objects.items()):
                x, y, z = entry.pose_map
                centroid = Vector3(x=x, y=y, z=z)
                # sphere marker id and text marker id (ensure unique ids)
                sphere_id = idx * 2
                text_id = idx * 2 + 1
                sphere_marker = self.create_centroid_marker(entry.name, centroid, int(obj_id.split('_')[-1]), sphere_id)
                marker_array.markers.append(sphere_marker)

                # create a text marker above the sphere
                text_marker = Marker()
                text_marker.header.frame_id = self.map_frame
                text_marker.header.stamp = self.get_clock().now().to_msg()
                text_marker.id = text_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = float(x)
                text_marker.pose.position.y = float(y)
                text_marker.pose.position.z = float(z + 0.15)
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.z = 0.08
                h = abs(hash(entry.name))
                text_marker.color.r = ((h >> 0) & 0xFF) / 255.0
                text_marker.color.g = ((h >> 8) & 0xFF) / 255.0
                text_marker.color.b = ((h >> 16) & 0xFF) / 255.0
                text_marker.color.a = 1.0
                text_marker.text = f"{entry.name}"
                marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)

    def export_callback(self):
        """Periodically export detections to CSV."""
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
        
        # Publish centroid markers
        self.publish_centroids()

    def shutdown_callback(self):
        """Export final results on shutdown."""
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
