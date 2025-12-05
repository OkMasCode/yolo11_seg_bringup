import rclpy
from rclpy.node import Node
from yolo11_seg_interfaces.msg import SemanticObjectArray


class SemanticMapPrinter(Node):
    def __init__(self):
        super().__init__('semantic_map_printer')
        # Subscribe to the semantic map topic and print all detections in each message.
        self.subscription = self.create_subscription(
            SemanticObjectArray,
            '/yolo/semantic_map',
            self.listener_callback,
            10,
        )

    def listener_callback(self, msg: SemanticObjectArray):
        for obj in msg.objects:
            x, y, z = obj.pose_map.x, obj.pose_map.y, obj.pose_map.z
            similarity = obj.similarity
            self.get_logger().info(
                f"name={obj.name}, coords=({x:.3f}, {y:.3f}, {z:.3f}), similarity={similarity:.3f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapPrinter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()