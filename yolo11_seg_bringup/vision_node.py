from rclpy.node import Node

# -------------------- CLASS ------------------- #

class VisionNode(Node):
    """ Node that handles Vision computation """
    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info("VisionNode initialized")

        

