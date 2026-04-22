"""ROS2 node that periodically computes a scene embedding and raw prompt similarity."""

import json
import os
import threading
from time import perf_counter

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from yolo11_seg_interfaces.msg import Similarity

from .utils.siglip2_processor import SIGLIPProcessor


class SceneEmbeddingNode(Node):
    """Compute a scene-level SigLIP embedding every N frames and compare it to a prompt."""

    def __init__(self):
        super().__init__('scene_embedding_node')
        self.get_logger().info('Scene Embedding Node initialized')

        self.declare_parameter('image_topic', '/jackal/sensors/camera_0/color/image')
        self.declare_parameter(
            'scene_prompt_file',
            '/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/scene_prompt.json',
        )
        self.declare_parameter('CLIP_model_name', 'google/siglip2-large-patch16-384')
        self.declare_parameter('sample_every_n_frames', 15)
        self.declare_parameter('prompt_check_interval', 2.0)
        self.declare_parameter('scene_embedding_topic', '/vision/scene_embedding')
        self.declare_parameter('scene_similarity_topic', '/vision/scene_similarity_raw')

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.scene_prompt_file = str(self.get_parameter('scene_prompt_file').value)
        self.CLIP_model_name = str(self.get_parameter('CLIP_model_name').value)
        self.sample_every_n_frames = max(1, int(self.get_parameter('sample_every_n_frames').value))
        self.prompt_check_interval = float(self.get_parameter('prompt_check_interval').value)
        self.scene_embedding_topic = str(self.get_parameter('scene_embedding_topic').value)
        self.scene_similarity_topic = str(self.get_parameter('scene_similarity_topic').value)

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Loading SigLIP model on device: {self.device}')
        self.clip = SIGLIPProcessor(device=self.device, model_name=self.CLIP_model_name)

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
        self.scene_embedding_pub = self.create_publisher(Float64MultiArray, self.scene_embedding_topic, 10)
        self.scene_similarity_pub = self.create_publisher(Similarity, self.scene_similarity_topic, 10)

        self.frame_count = 0
        self.prompt_lock = threading.Lock()
        self.current_prompt_text = None
        self.current_prompt_embedding = None
        self.last_scene_embedding = None
        self.last_scene_similarity = None

        self._load_scene_prompt()
        self.prompt_timer = self.create_timer(self.prompt_check_interval, self._load_scene_prompt)
        self.get_logger().info(
            f'Sampling every {self.sample_every_n_frames} frames from {self.image_topic}; '
            f'publishing embedding on {self.scene_embedding_topic} and raw similarity on {self.scene_similarity_topic}'
        )

    def image_callback(self, image_msg: Image):
        self.frame_count += 1
        if self.frame_count % self.sample_every_n_frames != 0:
            return

        frame_start = perf_counter()
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed converting RGB image to OpenCV frame: {exc}')
            return

        with self.prompt_lock:
            prompt_embedding = None if self.current_prompt_embedding is None else self.current_prompt_embedding.copy()
            prompt_text = self.current_prompt_text

        if prompt_embedding is None:
            self.get_logger().warn('No scene prompt embedding available yet; skipping scene similarity computation.')
            return

        scene_embedding = self.clip.encode_image(cv_bgr)
        if scene_embedding is None:
            self.get_logger().warn('Scene embedding computation returned no result.')
            return

        similarity_raw = self.clip.compute_match_logit(scene_embedding, prompt_embedding)
        self.last_scene_embedding = np.asarray(scene_embedding, dtype=np.float64)
        self.last_scene_similarity = float(similarity_raw)

        embedding_msg = Float64MultiArray()
        embedding_msg.data = [float(value) for value in self.last_scene_embedding.tolist()]
        similarity_msg = Similarity()
        similarity_msg.header = image_msg.header
        similarity_msg.similarity = float(similarity_raw)

        self.scene_embedding_pub.publish(embedding_msg)
        self.scene_similarity_pub.publish(similarity_msg)

        elapsed_ms = (perf_counter() - frame_start) * 1000.0
        self.get_logger().info(
            f"[scene_embedding] frame={self.frame_count} prompt='{prompt_text}' "
            f'raw_similarity={similarity_raw:.18e} elapsed_ms={elapsed_ms:.2f}'
        )

    def _load_scene_prompt(self):
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
        with self.prompt_lock:
            if prompt_key == self.current_prompt_text:
                return

        prompt_payload = prompt_value
        if isinstance(prompt_value, str):
            prompt_payload = self.clip.build_prompt_list(prompt_value)
        elif isinstance(prompt_value, list):
            expanded_prompts = []
            for item in prompt_value:
                expanded_prompts.extend(self.clip.build_prompt_list(item))
            prompt_payload = expanded_prompts

        prompt_embedding = self.clip.encode_text(prompt_payload)
        if prompt_embedding is None:
            self.get_logger().warn('Scene prompt encoding returned no embedding')
            return

        with self.prompt_lock:
            self.current_prompt_text = prompt_key
            self.current_prompt_embedding = np.asarray(prompt_embedding, dtype=np.float64)

        self.get_logger().info(f"Updated scene prompt from '{self.scene_prompt_file}': {prompt_key}")

    def _extract_prompt_value(self, data):
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


def main(args=None):
    rclpy.init(args=args)
    node = SceneEmbeddingNode()
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