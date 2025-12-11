import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import tf2_ros
from geometry_msgs.msg import Vector3
from rcl_interfaces.msg import SetParametersResult

from yolo11_seg_bringup.rgb_mapper import SemanticObjectMap
from yolo11_seg_interfaces.msg import DetectedObject, SemanticObjectArray, SemanticObject

# -------------------- NODE -------------------- #

class PointCloudMapperNode(Node):

    # ------------- Initialization ------------- #

    def __init__(self):
        super().__init__('pointcloud_mapper_node')

        # ============= Parameters ============= #

        self.declare_parameter('detection_message', '/yolo/detections')
        # In RGB case, we treat centroids as pixel coordinates; TF frames are unused
        self.declare_parameter('map_frame', 'camera_frame')
        self.declare_parameter('camera_frame', 'camera_frame')
        self.declare_parameter('semantic_map_topic', '/yolo/semantic_map')
        self.declare_parameter("output_dir", "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/")
        self.declare_parameter('export_interval', 5.0)
        self.declare_parameter('load_map_on_start', False)
        self.declare_parameter('input_map_file', 'map.json')
        # Pixel threshold for merging detections (RGB mode)
        self.declare_parameter('pixel_merge_threshold', 120.0)

        self.dm_topic = self.get_parameter('detection_message').value
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.semantic_map_topic = self.get_parameter('semantic_map_topic').value
        self.output_dir = self.get_parameter("output_dir").value

        self.export_interval = float(self.get_parameter('export_interval').value)
        self.load_map_on_start = self.get_parameter('load_map_on_start').value
        self.input_map_file = self.get_parameter('input_map_file').value
        self.pixel_merge_threshold = float(self.get_parameter('pixel_merge_threshold').value)

        # =========== Initialization =========== #

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Enable RGB mode in the mapper (pixel-frame, no TF transform)
        self.semantic_map = SemanticObjectMap(self.tf_buffer, self, use_pixel_frame=True)

        # Allow runtime parameter updates (e.g., ros2 param set)
        self.add_on_set_parameters_callback(self._on_set_params)

        # Load existing map if enabled
        if self.load_map_on_start:
            self.semantic_map.load_from_json(
                directory_path=self.output_dir,
                file=self.input_map_file,
            )
            print(f"Loaded existing semantic map from {self.input_map_file}")
            print(f"Current number of objects in map: {len(self.semantic_map.objects)}")

        qos_sensor = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.dm_sub = self.create_subscription(DetectedObject, self.dm_topic, self.detection_callback, qos_profile=qos_sensor)
        self.map_pub = self.create_publisher(SemanticObjectArray, self.semantic_map_topic, 10)
        
        self.export_timer = self.create_timer(self.export_interval, self.export_callback)

        self.lock = threading.Lock()

        self.get_logger().info(
            f"RGB Mapper Node initialized. Subscribing to {self.dm_topic}, merging threshold {self.pixel_merge_threshold}px, output to {self.output_dir}"
        )

    # ---------------- Callbacks --------------- #

    def detection_callback(self, msg: DetectedObject):
        """
        Process incoming DetectedObject message data.
        """
        try:
            with self.lock:
                # Extract fields from DetectedObject message
                class_name = msg.object_name
                instance_id = msg.object_id
                centroid = msg.centroid
                timestamp = msg.timestamp
                embedding = msg.embedding
                text_embedding = msg.text_embedding

                # Create a unique object ID for this particular detection
                object_id = (
                    f"{class_name}_inst{instance_id}_"
                    f"{timestamp.sec}_{timestamp.nanosec}"
                )

                # Call the method of the mapper node
                self.semantic_map.add_detection(
                    object_name = class_name,
                    object_id = object_id,
                    pose_in_camera = centroid, # Geometric position in the camera frame
                    detection_stamp = timestamp,
                    camera_frame = self.camera_frame, # Camera frame ID
                    fixed_frame = self.map_frame, # Fixed map frame ID
                    # In RGB mode, centroids are pixel coordinates; use pixel threshold
                    distance_threshold = self.pixel_merge_threshold,
                    embeddings = embedding,
                    goal_embedding = text_embedding
                )

                self.publish_semantic_map()
                
        except Exception as e:
            self.get_logger().error(f"Error processing DetectedObject: {e}")

    def _on_set_params(self, params):
        result = SetParametersResult()
        result.successful = True
        result.reason = ''

        try:
            for p in params:
                if p.name == 'pixel_merge_threshold':
                    val = float(p.value)
                    if val <= 0:
                        result.successful = False
                        result.reason = 'pixel_merge_threshold must be > 0'
                        return result
                    self.pixel_merge_threshold = val
                    self.get_logger().info(f"Updated pixel_merge_threshold to {self.pixel_merge_threshold}px")
        except Exception as ex:
            result.successful = False
            result.reason = str(ex)

        return result

    def publish_semantic_map(self):
        """
        Publish the current semantic object map.
        """
        if not self.semantic_map.objects:
            return  # No objects to publish
        
        msg = SemanticObjectArray()
        msg.objects = []
        append = msg.objects.append

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
            # Publish image_embedding if field exists in SemanticObject
            if hasattr(obj, 'image_embedding'):
                obj.image_embedding = entry.image_embedding.tolist() if entry.image_embedding is not None else []
            append(obj)

        self.map_pub.publish(msg)

    def export_callback(self):
        """
        Export the current semantic map stored to a json file
        """
        with self.lock:
            try:
                self.semantic_map.export_to_json(
                    directory_path=self.output_dir,
                    file='map.json'
                )
            except Exception as e:
                self.get_logger().error(f"Error exporting semantic map: {e}")

    def shutdown_callback(self):
        """
        Final export of the semantic map to a json file during node shutdown.
        """
        with self.lock:
            try:
                self.semantic_map.export_to_json(
                    directory_path=self.output_dir,
                    file="map_final.json"
                )
            except Exception as e:
                self.get_logger().error(f"Final export failed: {e}")

# -------------------- MAIN -------------------- #

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