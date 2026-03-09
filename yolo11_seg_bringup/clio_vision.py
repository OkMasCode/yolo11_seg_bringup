import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import json
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # --- 1. Load Models ---
        self.get_logger().info("Loading YOLO and SigLIP models...")
        self.yolo = YOLO('yolov8n-seg.pt')  # Use a segmentation model
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.get_logger().info("Models loaded!")

        # --- 2. ROS Setup ---
        self.bridge = CvBridge()
        
        # Input: RGB Image
        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
            
        # Output A: Semantic Image (ID per pixel)
        self.pub_sem = self.create_publisher(Image, '/perception/semantic_image', 10)
        
        # Output B: Embeddings (ID -> Vector)
        self.pub_embed = self.create_publisher(String, '/perception/embeddings', 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # --- 3. Run Detection ---
        results = self.yolo(cv_image, verbose=False)[0]
        
        if results.masks is None:
            return # No objects found

        # Prepare outputs
        h, w = cv_image.shape[:2]
        semantic_img = np.zeros((h, w), dtype=np.int32) # 32-bit signed int for IDs
        embeddings_dict = {}
        
        # --- 4. Process Each Detection ---
        for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
            # Create a unique ID for this frame (avoid 0, it's background)
            # In a real tracker, you'd use the tracker ID. Here we use index + 1 for simplicity.
            obj_id = int(i + 1)
            
            # Fill the mask on the semantic image
            # mask.data is a torch tensor, convert to numpy and resize if needed
            m = mask.data[0].cpu().numpy()
            m = cv2.resize(m, (w, h)).astype(bool)
            semantic_img[m] = obj_id

            # Crop object for SigLIP
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = cv_image[y1:y2, x1:x2]
            
            if crop.size == 0: continue

            # Extract Embedding
            inputs = self.processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get the vector and normalize it
            embedding = outputs.pooler_output[0].cpu().numpy().tolist()
            embeddings_dict[str(obj_id)] = embedding

        # --- 5. Publish Synchronized Data ---
        timestamp = msg.header.stamp
        
        # Publish Image
        sem_msg = self.bridge.cv2_to_imgmsg(semantic_img, encoding="32SC1")
        sem_msg.header.stamp = timestamp
        sem_msg.header.frame_id = msg.header.frame_id
        self.pub_sem.publish(sem_msg)
        
        # Publish Metadata
        embed_msg = String()
        embed_msg.data = json.dumps(embeddings_dict)
        self.pub_embed.publish(embed_msg)
        
        self.get_logger().info(f"Published {len(embeddings_dict)} objects.")

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()