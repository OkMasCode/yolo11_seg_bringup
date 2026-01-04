from rclpy.node import Node
import rclpy

from yolo11_seg_interfaces.msg import SemanticObjectArray
from yolo11_seg_interfaces.msg import Vector2

import json

class GoalDetector(Node):
    def __init__(self):
        super().__init__('goal_detector')
        # Initialization code for GoalDetector goes here
        self.get_logger().info("GoalDetector node has been initialized.")

        self.map_sub = self.create_subscription(
            SemanticObjectArray,
            '/vision/semantic_map',
            self.map_callback,
            10
        )

        self.goal_pub = self.create_publisher(
            Vector2,
            '/robot/goal_position',
            10
        )

        self.file_path = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json"
        self.goal_name = ""

    def map_callback(self, msg: SemanticObjectArray):
        sim = 0.0
        with open(self.file_path, 'r') as f:
            robot_goal = json.load(f)
            self.goal_name = robot_goal["goal"]
        for obj in msg.objects:
            if obj.similarity > sim and obj.name == self.goal_name:
                sim = obj.similarity
                goal_x = obj.pose_map.x
                goal_y = obj.pose_map.y
        
        goal_msg = Vector2()
        goal_msg.x = goal_x
        goal_msg.y = goal_y
        self.get_logger().info(f"Detected goal '{self.goal_name}' at position: ({goal_x}, {goal_y}) with similarity {sim}")
        self.goal_pub.publish(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GoalDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()