import rclpy
from rclpy.node import Node
import torch
import clip
import numpy as np

# TODO: Replace 'my_robot_interfaces' with your actual package name
from yolo11_seg_interfaces.msg import DetectedObject

class SemanticSearchNode(Node):
    def __init__(self):
        super().__init__('semantic_search_node')

        # --- CONFIGURATION ---
        self.target_prompt = "a bed with red stripes"
        self.similarity_threshold = 0.20  # Filter out low matches if needed
        # ---------------------

        self.get_logger().info("Loading CLIP model for text encoding...")
        
        # Load CLIP (using CPU is fine for just text encoding at startup)
        # If you have GPU memory to spare, change 'cpu' to 'cuda'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)

        # 1. PRE-COMPUTE TEXT EMBEDDING
        # We do this ONCE here so we don't waste time doing it for every detection
        self.text_features = self.encode_text_prompt(self.target_prompt)
        self.get_logger().info(f"Searching for: '{self.target_prompt}'")

        # 2. CREATE SUBSCRIBER
        self.subscription = self.create_subscription(
            DetectedObject,
            '/yolo/detections',  # Change to your actual topic name
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def encode_text_prompt(self, text):
        """Encodes text into a normalized CLIP embedding tensor."""
        with torch.no_grad():
            text_token = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_token)
            # Normalize the vector (crucial for cosine similarity)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def listener_callback(self, msg):
        # 3. READ INCOMING EMBEDDING
        # msg.embedding is a list of float32, convert to Tensor
        obj_embedding = torch.tensor(msg.embedding).to(self.device)
        
        # Reshape to [1, 512] and normalize
        obj_embedding = obj_embedding.unsqueeze(0)
        obj_embedding /= obj_embedding.norm(dim=-1, keepdim=True)

        # 4. COMPUTE SIMILARITY (Dot Product)
        # Since vectors are normalized, dot product == cosine similarity
        similarity = (obj_embedding @ self.text_features.T).item()
        
        # Convert to percentage
        sim_percent = similarity * 100

        # 5. OUTPUT RESULT
        # Format: class, id, centroid coordinates = similarity percentage
        output_str = (
            f"Class: {msg.object_name}, "
            f"ID: {msg.object_id}, "
            f"Centroid: ({msg.centroid.x:.2f}, {msg.centroid.y:.2f}, {msg.centroid.z:.2f}) = "
            f"{sim_percent:.2f}%"
        )

        # Print to console (stdout)
        print(output_str)
        
        # Optional: Log to ROS2 logger as well
        # self.get_logger().info(output_str)

def main(args=None):
    rclpy.init(args=args)
    node = SemanticSearchNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()