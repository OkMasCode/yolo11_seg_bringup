import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import json
import os
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from .utils.siglip2_processor import SIGLIPProcessor

class SiglipHeatmapPublisher(Node):
    def __init__(self):
        super().__init__('siglip_heatmap_publisher')

        # --- Parameters ---
        self.declare_parameter('image_topic', '/jackal/sensors/camera_0/color/image')
        self.declare_parameter(
            'scene_prompt_file',
            '/workspaces/ros2_ws/src/yolo11_seg_bringup/config/scene_prompt.json',
        )
        self.declare_parameter('prompt_check_interval', 2.0)
        self.declare_parameter('clip_model_name', 'google/siglip2-large-patch16-384')

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.scene_prompt_file = str(self.get_parameter('scene_prompt_file').value)
        self.prompt_check_interval = float(self.get_parameter('prompt_check_interval').value)
        self.clip_model_name = str(self.get_parameter('clip_model_name').value)
        
        self.bridge = CvBridge()
        
        # --- Subscribers & Publishers ---
        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile=qos_sensor,
        )
        
        qos_heatmap = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher_ = self.create_publisher(Image, 'siglip_heatmap', qos_profile=qos_heatmap)
        
        # --- Model Initialization ---
        self.get_logger().info("Loading SigLIP model...")
        self.processor = SIGLIPProcessor(device="cuda", model_name=self.clip_model_name)
        self.get_logger().info("Model loaded!")

        # Default prompt
        self.search_prompt = "a dog"

        # Timer to update the text prompt from the JSON file
        self._load_scene_prompt()
        self.prompt_timer = self.create_timer(self.prompt_check_interval, self._load_scene_prompt)

    def _load_scene_prompt(self):
        """Reads the target object text from a JSON file."""
        if not os.path.exists(self.scene_prompt_file):
            self.get_logger().warn(
                f"scene prompt file not found: '{self.scene_prompt_file}'",
                throttle_duration_sec=10.0,
            )
            return

        try:
            with open(self.scene_prompt_file, 'r', encoding='utf-8') as file_handle:
                data = json.load(file_handle)
        except Exception as exc:
            self.get_logger().error(f'Failed reading scene prompt file: {exc}')
            return

        prompt_value = self._extract_prompt_value(data)
        if prompt_value is None:
            self.get_logger().warn(
                f"No valid scene prompt found in '{self.scene_prompt_file}'",
                throttle_duration_sec=5.0,
            )
            return

        prompt_key = self._prompt_key(prompt_value)
        
        # Only update if the prompt has changed
        if prompt_key == self.search_prompt:
            return

        self.search_prompt = prompt_key
        self.get_logger().info(f"Updated scene prompt from '{self.scene_prompt_file}': {self.search_prompt}")

    def _extract_prompt_value(self, data):
        """Helper function to parse the JSON prompt file safely."""
        if not isinstance(data, dict):
            return None

        for key in ('scene_prompt', 'scene_prompts', 'prompt', 'clip_prompts'):
            if key not in data:
                continue
            value = data.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
            elif isinstance(value, list):
                cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
                if cleaned:
                    return cleaned
        return None

    def _prompt_key(self, prompt_value):
        if isinstance(prompt_value, list):
            return '\n'.join(prompt_value)
        return str(prompt_value)

    def image_callback(self, msg):
        """Processes incoming camera frames and publishes the heatmap."""
        
        # 1. Convert ROS Image message to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # 2. Generate the heatmap using the new FindAnything logic
        try:
            # We pass the image and the raw text prompt string directly to the processor
            blended_image, _ = self.processor.generate_findanything_heatmap(
                cv_image, 
                self.search_prompt
            )
        except Exception as e:
            self.get_logger().error(f"Failed to generate heatmap: {e}")
            return

        # 3. Convert back to ROS Image message and publish
        try:
            ros_image_msg = self.bridge.cv2_to_imgmsg(blended_image, encoding="bgr8")
            
            # Keep the original timestamp and frame_id to maintain synchronization in ROS tools like RViz
            ros_image_msg.header = msg.header 
            
            self.publisher_.publish(ros_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SiglipHeatmapPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()