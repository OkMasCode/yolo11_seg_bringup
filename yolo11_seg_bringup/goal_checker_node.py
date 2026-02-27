import json
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor

from yolo11_seg_interfaces.msg import SemanticObjectArray

# -------------------- NODE -------------------- #

class GoalCheckerNode(Node):

    # ------------- Initialization ------------- #

    def __init__(self):
        super().__init__('goal_checker_node')

        # ============= Parameters ============= #

        self.declare_parameter('semantic_map_topic', '/vision/semantic_map')
        self.declare_parameter('goal_flag_topic', '/vision/goal_reached')
        self.declare_parameter('goal_position_topic', '/vision/goal_position')
        self.declare_parameter('similarity_threshold', 5.0)
        self.declare_parameter('command_file_path', '/workspaces/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json')

        self.semantic_map_topic = self.get_parameter('semantic_map_topic').value
        self.goal_flag_topic = self.get_parameter('goal_flag_topic').value
        self.goal_position_topic = self.get_parameter('goal_position_topic').value
        self.similarity_threshold = float(self.get_parameter('similarity_threshold').value)
        self.command_file_path = self.get_parameter('command_file_path').value

        # =========== Initialization =========== #

        self.goal_object = None
        self.goal_class = None
        self.goal_reached = False

        # Load goal from command file
        self.load_goal_from_command_file()

        if self.goal_class is None:
            self.get_logger().warn("No goal object loaded from command file")

        # Subscribe to semantic object map
        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.semantic_map_sub = self.create_subscription(
            SemanticObjectArray,
            self.semantic_map_topic,
            self.semantic_map_callback,
            qos_profile=qos_sensor
        )

        # Publisher for goal reached flag
        self.goal_flag_pub = self.create_publisher(Bool, self.goal_flag_topic, 10)
        
        # Publisher for goal pose
        self.goal_position_pub = self.create_publisher(PoseStamped, self.goal_position_topic, 10)

        self.lock = threading.Lock()
        self.current_goal_position = None

        self.get_logger().info(
            f"Goal Checker Node initialized. "
            f"Subscribing to {self.semantic_map_topic}, "
            f"publishing to {self.goal_flag_topic} and {self.goal_position_topic}, "
            f"threshold: {self.similarity_threshold}, "
            f"goal: {self.goal_class}"
        )

    # ------------ Helper Methods ------------ #

    def load_goal_from_command_file(self):
        """
        Load goal information from the robot command JSON file.
        """
        try:
            if not Path(self.command_file_path).exists():
                self.get_logger().warn(f"Command file not found at {self.command_file_path}")
                return

            with open(self.command_file_path, 'r') as f:
                command_data = json.load(f)

            # Extract goal from command file
            self.goal_object = command_data.get('goal', None)
            
            if self.goal_object:
                self.goal_class = self.goal_object.lower()
                self.get_logger().info(f"Loaded goal: {self.goal_class}")
            else:
                self.get_logger().warn("No 'goal' field found in command file")

        except Exception as e:
            self.get_logger().error(f"Error loading goal from command file: {e}")

    # ------------- Callbacks ------------- #

    def semantic_map_callback(self, msg: SemanticObjectArray):
        """
        Process incoming SemanticObjectArray messages.
        Check if any detected objects match the goal class and exceed similarity threshold.
        """
        try:
            with self.lock:
                if self.goal_class is None:
                    return

                goal_seen = False
                matching_objects = []
                valid_goal_candidates = []

                # Check all objects in the semantic map
                for obj in msg.objects:
                    object_class = obj.name.lower()

                    # Check if this object matches the goal class
                    if object_class == self.goal_class:
                        matching_objects.append({
                            'id': obj.object_id,
                            'name': obj.name,
                            'similarity': obj.similarity,
                            'pose_map': (obj.pose_map.x, obj.pose_map.y, obj.pose_map.z)
                        })

                        # Keep only candidates that exceed threshold
                        if obj.similarity >= self.similarity_threshold:
                            valid_goal_candidates.append(obj)

                # Select and publish the best valid goal candidate (highest similarity)
                if valid_goal_candidates:
                    best_goal_obj = max(valid_goal_candidates, key=lambda goal_obj: goal_obj.similarity)
                    goal_seen = True
                    self.current_goal_position = (
                        best_goal_obj.pose_map.x,
                        best_goal_obj.pose_map.y,
                        best_goal_obj.pose_map.z
                    )

                    pose_msg = PoseStamped()
                    pose_msg.header = best_goal_obj.timestamp
                    pose_msg.pose.position.x = best_goal_obj.pose_map.x
                    pose_msg.pose.position.y = best_goal_obj.pose_map.y
                    pose_msg.pose.position.z = best_goal_obj.pose_map.z
                    pose_msg.pose.orientation.x = 0.0
                    pose_msg.pose.orientation.y = 0.0
                    pose_msg.pose.orientation.z = 0.0
                    pose_msg.pose.orientation.w = 1.0
                    self.goal_position_pub.publish(pose_msg)

                    self.get_logger().info(
                        f"GOAL DETECTED: {best_goal_obj.name} "
                        f"(similarity: {best_goal_obj.similarity:.4f}, threshold: {self.similarity_threshold}) "
                        f"at position [x={best_goal_obj.pose_map.x:.3f}, y={best_goal_obj.pose_map.y:.3f}, z={best_goal_obj.pose_map.z:.3f}]"
                    )

                # Publish the goal reached flag
                flag_msg = Bool()
                flag_msg.data = goal_seen
                self.goal_flag_pub.publish(flag_msg)

                # Update state
                if goal_seen and not self.goal_reached:
                    self.goal_reached = True
                    print(f"\n{'='*60}")
                    print(f"GOAL HAS BEEN SEEN: {self.goal_class.upper()}")
                    print(f"{'='*60}\n")
                elif not goal_seen and self.goal_reached:
                    self.goal_reached = False
                    print(f"\n{'='*60}")
                    print(f"GOAL LOST: {self.goal_class.upper()}")
                    print(f"{'='*60}\n")

                # Log matching objects (even if below threshold)
                if matching_objects and not goal_seen:
                    self.get_logger().debug(
                        f"Found {len(matching_objects)} {self.goal_class} object(s) "
                        f"but below similarity threshold {self.similarity_threshold}"
                    )

        except Exception as e:
            self.get_logger().error(f"Error processing semantic map: {e}")

    def shutdown_callback(self):
        """
        Called during node shutdown.
        """
        self.get_logger().info("Goal Checker Node shutting down...")

# -------------------- MAIN -------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = GoalCheckerNode()

    # Use MultiThreadedExecutor to allow proper concurrent operation
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

if __name__ == "__main__":
    main()
