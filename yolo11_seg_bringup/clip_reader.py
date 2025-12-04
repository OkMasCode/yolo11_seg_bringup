import rclpy
from rclpy.node import Node
import numpy as np
from yolo11_seg_interfaces.msg import DetectedObject

class SimilarityNode(Node):
    def __init__(self):
        super().__init__('similarity_calculator')
        self.subscription = self.create_subscription(
            DetectedObject,
            '/yolo/detections',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        # 1. Extract Vectors (Msg arrays -> Numpy arrays)
        img_vec = np.array(msg.embedding, dtype=np.float32)
        txt_vec = np.array(msg.text_embedding, dtype=np.float32)

        # 2. Compute Similarity
        # Since Node 1 already normalized them, we just dot product.
        similarity = np.dot(img_vec, txt_vec)
        
        # 3. Output
        print(f"Object {msg.object_id} Similarity: {similarity * 100:.2f}%")


def main(args=None):
    rclpy.init(args=args)
    node = SimilarityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()