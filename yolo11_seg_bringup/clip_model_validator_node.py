import threading
import json
import os

import cv2
import numpy as np
import open_clip
import rclpy
import torch
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class ClipModelValidatorNode(Node):
    """
    Minimal node for CLIP model validation.
    Inputs:
      - Image topic
            - Prompt from JSON file
    Output:
      - Similarity topic (std_msgs/Float32)
    """

    def __init__(self):
        super().__init__('clip_model_validator_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('similarity_topic', '/clip/similarity')
        self.declare_parameter('clip_model_name', 'ViT-SO400M-14-SigLIP2-378')
        self.declare_parameter('clip_pretrained', 'webli')
        self.declare_parameter('prompt_file', '/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/clip_prompt.json')
        self.declare_parameter('prompt_check_interval', 30.0)

        self.image_topic = self.get_parameter('image_topic').value
        self.similarity_topic = self.get_parameter('similarity_topic').value
        self.clip_model_name = self.get_parameter('clip_model_name').value
        self.clip_pretrained = self.get_parameter('clip_pretrained').value
        self.prompt_file = self.get_parameter('prompt_file').value
        self.prompt_check_interval = float(self.get_parameter('prompt_check_interval').value)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Loading CLIP model: {self.clip_model_name} ({self.clip_pretrained}) on {self.device}')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.clip_model_name,
            pretrained=self.clip_pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        self.model.eval()

        model_image_size = getattr(getattr(self.model, 'visual', None), 'image_size', None)
        if isinstance(model_image_size, (tuple, list)) and len(model_image_size) > 0:
            self.clip_image_size = int(model_image_size[0])
        elif model_image_size is not None:
            self.clip_image_size = int(model_image_size)
        else:
            self.clip_image_size = 224

        self.bridge = CvBridge()
        self.state_lock = threading.Lock()
        self.current_prompt = None
        self.current_text_embedding = None

        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile=qos_sensor)
        self.sim_pub = self.create_publisher(Float32, self.similarity_topic, 10)

        self._load_prompt_from_json()
        self.prompt_timer = self.create_timer(self.prompt_check_interval, self._load_prompt_from_json)

        self.get_logger().info('ClipModelValidatorNode ready.')
        self.get_logger().info(f'Input image topic: {self.image_topic}')
        self.get_logger().info(f'Prompt file: {self.prompt_file}')
        self.get_logger().info(f'Output similarity topic: {self.similarity_topic}')
        self.get_logger().info(f'CLIP input size: {self.clip_image_size}')

    def _encode_text(self, text):
        if text is None:
            return None

        if isinstance(text, str):
            text = [text.strip()] if text.strip() else []
        elif isinstance(text, list):
            text = [item.strip() for item in text if isinstance(item, str) and item.strip()]
        else:
            return None

        if not text:
            return None

        with torch.no_grad():
            try:
                tokens = self.tokenizer(text).to(self.device)
            except AttributeError:
                if hasattr(self.tokenizer, 'tokenizer'):
                    hf_tokenizer = self.tokenizer.tokenizer
                    context_length = getattr(self.tokenizer, 'context_length', 77)
                    encoded = hf_tokenizer(
                        text,
                        padding='max_length',
                        truncation=True,
                        max_length=context_length,
                        return_tensors='pt',
                    )
                    tokens = encoded['input_ids'].to(self.device)
                else:
                    raise

            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            ensemble_feature = features.mean(dim=0, keepdim=True)
            ensemble_feature = ensemble_feature / ensemble_feature.norm(dim=-1, keepdim=True)

        return ensemble_feature.squeeze(0).detach().cpu().numpy()

    def _encode_image(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        with torch.no_grad():
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).detach().cpu().numpy()

    def _load_prompt_from_json(self):
        if not os.path.exists(self.prompt_file):
            self.get_logger().warn(f'Prompt file not found: {self.prompt_file}', throttle_duration_sec=10.0)
            return

        try:
            with open(self.prompt_file, 'r') as file_handle:
                data = json.load(file_handle)

            prompt = data.get('clip_prompt', None)
            if not isinstance(prompt, str) or not prompt.strip():
                self.get_logger().warn('Prompt file has no valid "clip_prompt" string.', throttle_duration_sec=10.0)
                return

            prompt = prompt.strip()
            with self.state_lock:
                if prompt == self.current_prompt:
                    return

                text_embedding = self._encode_text(prompt)
                if text_embedding is None:
                    self.current_text_embedding = None
                    self.get_logger().warn('Failed to encode prompt from file.')
                    return

                self.current_prompt = prompt
                self.current_text_embedding = text_embedding

            self.get_logger().info(f'Updated prompt from JSON: {prompt}')

        except Exception as exc:
            self.get_logger().error(f'Failed reading prompt file: {exc}')

    def image_callback(self, msg: Image):
        with self.state_lock:
            text_embedding = None if self.current_text_embedding is None else self.current_text_embedding.copy()
            current_prompt = self.current_prompt

        if text_embedding is None:
            self.get_logger().warn('Waiting for prompt embedding from prompt file.', throttle_duration_sec=5.0)
            return

        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image_embedding = self._encode_image(cv_bgr)
            if image_embedding is None:
                self.get_logger().warn('Image embedding failed.')
                return
            probability = self._compute_match_probability(image_embedding, text_embedding)

            sim_msg = Float32()
            sim_msg.data = float(probability)
            self.sim_pub.publish(sim_msg)
            self.get_logger().info(f'Match probability ({current_prompt}): {probability:.2f}%')

        except Exception as exc:
            self.get_logger().error(f'Failed to process image/prompt pair: {exc}')

    @staticmethod
    def _safe_sigmoid(value):
        clipped = np.clip(value, -60.0, 60.0)
        return float(1.0 / (1.0 + np.exp(-clipped)))

    def _compute_match_probability(self, image_embedding, text_embedding):
        image_vec = np.asarray(image_embedding, dtype=np.float32).reshape(-1)
        text_vec = np.asarray(text_embedding, dtype=np.float32).reshape(-1)

        image_norm = np.linalg.norm(image_vec)
        text_norm = np.linalg.norm(text_vec)
        if image_norm == 0.0 or text_norm == 0.0:
            return 0.0

        image_vec = image_vec / image_norm
        text_vec = text_vec / text_norm

        dot_product = float(np.dot(image_vec, text_vec))

        with torch.no_grad():
            logit_scale = float(self.model.logit_scale.exp().item()) if hasattr(self.model, 'logit_scale') else 1.0
            logit_bias = float(self.model.logit_bias.item()) if hasattr(self.model, 'logit_bias') else 0.0

        logits = (dot_product * logit_scale) + logit_bias
        probability = self._safe_sigmoid(logits) * 100.0
        return float(probability)


def main(args=None):
    rclpy.init(args=args)
    node = ClipModelValidatorNode()
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
