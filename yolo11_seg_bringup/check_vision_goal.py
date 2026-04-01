import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Bool

# Import the message type published by your vision node
from yolo11_seg_interfaces.msg import DetectedObjectV3Array

class GoalCheckerNode(Node):
    def __init__(self):
        super().__init__('goal_checker_node')

        # --- Parameters ---
        self.declare_parameter('detections_topic', '/vision/detections')
        self.declare_parameter('goal_flag_topic', '/vision/goal_reached')
        self.declare_parameter('similarity_threshold', 15.0)
        self.declare_parameter('command_file_path', '/workspaces/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json')

        self.detections_topic = self.get_parameter('detections_topic').value
        self.goal_flag_topic = self.get_parameter('goal_flag_topic').value
        self.similarity_threshold = float(self.get_parameter('similarity_threshold').value)
        self.command_file_path = self.get_parameter('command_file_path').value

        # --- Initialization ---
        self.goal_class = None
        self.goal_reached = False
        
        # Load the target object name
        self.load_goal_from_command_file()

        # --- Publishers and Subscribers ---
        self.goal_flag_pub = self.create_publisher(Bool, self.goal_flag_topic, 10)
        
        # Match the Best Effort QoS profile used by the vision node
        qos_profile = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.detections_sub = self.create_subscription(
            DetectedObjectV3Array,
            self.detections_topic,
            self.detections_callback,
            qos_profile
        )

        self.get_logger().info(
            f"Goal Checker active. Looking for '{self.goal_class}' "
            f"with similarity > {self.similarity_threshold}"
        )

    def load_goal_from_command_file(self):
        """Loads the target goal name from the JSON command file using standard libraries."""
        try:
            if not Path(self.command_file_path).exists():
                self.get_logger().warn(f"Command file not found at {self.command_file_path}")
                return

            with open(self.command_file_path, 'r') as f:
                command_data = json.load(f)

            goal_object = command_data.get('goal', None)
            if goal_object:
                self.goal_class = goal_object.lower()
            else:
                self.get_logger().warn("No 'goal' found in command file.")
        except Exception as e:
            self.get_logger().error(f"Error loading command file: {e}")

    def detections_callback(self, msg: DetectedObjectV3Array):
        """Evaluates incoming detections using the pre-computed similarity score."""
        if not self.goal_class:
            return

        goal_seen = False

        # Iterate through all objects detected by the vision node in this frame
        for det in msg.detections:
            
            # 1. Check if the YOLO class matches our target goal
            if det.class_name.lower() == self.goal_class:
                
                # 2. Check if the vision node's computed similarity exceeds our threshold
                if det.similarity > self.similarity_threshold:
                    goal_seen = True
                    
                    self.get_logger().info(
                        f"*** GOAL DETECTED ***: {det.class_name} "
                        f"(Similarity: {det.similarity:.2f} > {self.similarity_threshold})"
                    )
                    break # Stop checking other objects since the goal is found

        # Publish the boolean result
        flag_msg = Bool()
        flag_msg.data = goal_seen
        self.goal_flag_pub.publish(flag_msg)

        # Print state changes to the terminal
        if goal_seen and not self.goal_reached:
            self.goal_reached = True
            print(f"\n{'='*60}\nGOAL HAS BEEN SEEN: {self.goal_class.upper()}\n{'='*60}\n")
        elif not goal_seen and self.goal_reached:
            self.goal_reached = False
            print(f"\n{'='*60}\nGOAL LOST: {self.goal_class.upper()}\n{'='*60}\n")

def main(args=None):
    rclpy.init(args=args)
    node = GoalCheckerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()