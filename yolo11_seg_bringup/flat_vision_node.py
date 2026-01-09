import os
import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

from .utils.clip_processor import CLIPProcessor


class FlatVisionNode(Node):
    """RGB-only vision node that mirrors vision_node logic without depth or publishers."""

    def __init__(self):
        super().__init__("flat_vision_node")
        self.get_logger().info("FlatVisionNode initialized")

        # Parameters
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("model_path", "/home/sensor/yolov8n-seg.engine")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.45)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("retina_masks", True)
        self.declare_parameter("CLIP_model_name", "ViT-B-16-SigLIP")
        self.declare_parameter("robot_command_file", "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json")
        self.declare_parameter("square_crop_scale", 1.2)
        self.declare_parameter("frame_skip", 1)
        self.declare_parameter("prompt_check_interval", 10.0)
        self.declare_parameter("annotated_topic", "/flat/annotated_image")

        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.retina_masks = bool(self.get_parameter("retina_masks").value)
        self.CLIP_model_name = self.get_parameter("CLIP_model_name").value
        self.robot_command_file = self.get_parameter("robot_command_file").value
        self.square_crop_scale = float(self.get_parameter("square_crop_scale").value)
        self.frame_skip = max(1, int(self.get_parameter("frame_skip").value))
        self.prompt_check_interval = float(self.get_parameter("prompt_check_interval").value)
        self.anno_topic = self.get_parameter("annotated_topic").value

        # Models
        self.get_logger().info(f"Loading YOLO: {self.model_path}")
        self.model = YOLO(self.model_path, task="segment")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on device: {self.device}")
        self.clip = CLIPProcessor(device=self.device, model_name=self.CLIP_model_name, pretrained="webli")

        self.bridge = CvBridge()
        self.frame_count = 0
        self.current_clip_prompt = None
        self.goal_text_embedding = None
        self.sync_lock = threading.Lock()

        self.CLASS_NAMES = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]

        # Subscriptions
        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.rgb_sub = self.create_subscription(Image, self.image_topic, self.rgb_callback, qos_profile=qos_sensor)

        # Publisher for annotated images
        self.vis_pub = self.create_publisher(Image, self.anno_topic, 10)

        # Prompt loader timer
        self._load_clip_prompt()
        self.command_timer = self.create_timer(self.prompt_check_interval, self._load_clip_prompt)

        self.get_logger().info("FlatVisionNode ready (no publishers; logs only)")

    # ------------ Utilities ------------ #

    def class_id_to_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id]
        return f"class_{class_id}"

    def _load_clip_prompt(self):
        """Load clip_prompt from robot_command.json and update text embedding if changed."""
        try:
            if not os.path.exists(self.robot_command_file):
                return

            with open(self.robot_command_file, "r") as f:
                data = json.load(f)

            clip_prompt = data.get("clip_prompt", None)
            if clip_prompt != self.current_clip_prompt:
                self.current_clip_prompt = clip_prompt
                self.goal_text_embedding = self.clip.encode_text(clip_prompt)
                self.get_logger().info(f"Updated Goal Ensemble: {clip_prompt}")
        except Exception as e:
            self.get_logger().error(f"Error loading robot_command.json: {e}")

    # ------------ Callbacks ------------ #

    def rgb_callback(self, rgb_msg: Image):
        with self.sync_lock:
            self.process_frame(rgb_msg)

    # ------------ Main Pipeline ------------ #

    def process_frame(self, rgb_msg: Image):
        self.frame_count += 1

        cv_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        height, width, _ = cv_bgr.shape

        results = self.model.track(
            source=cv_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            retina_masks=self.retina_masks,
            stream=False,
            verbose=False,
            persist=True,
            tracker="botsort.yaml",
        )
        res = results[0]
        if not hasattr(res, "masks") or len(res.boxes) == 0:
            return

        xyxy = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.arange(len(clss))
        masks_t = res.masks.data if hasattr(res, "masks") and res.masks is not None else None

        detections = []
        batch_queue = []
        do_clip_frame = (self.frame_count % self.frame_skip == 0)

        for i in range(len(xyxy)):
            entry = self.process_single_detection(
                idx=i,
                bbox=xyxy[i],
                class_id=clss[i],
                instance_id=ids[i],
                masks_t=masks_t,
                cv_bgr=cv_bgr,
                height=height,
                width=width,
                do_clip_frame=do_clip_frame,
            )
            if entry is None:
                continue
            detections.append(entry)
            if entry.get("crop") is not None:
                batch_queue.append(entry)

        if batch_queue:
            try:
                images = [item["crop"] for item in batch_queue]
                embeddings = self.clip.encode_images_batch(images)
                for i, emb in enumerate(embeddings):
                    batch_queue[i]["embedding"] = emb
                    batch_queue[i]["crop"] = None
            except Exception as e:
                self.get_logger().error(f"Batch CLIP inference failed: {e}")

        self.log_detections(detections)

        # Publish annotated image for visualization
        annotated_frame = res.plot()
        vis_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        vis_msg.header = rgb_msg.header
        self.vis_pub.publish(vis_msg)

    def process_single_detection(self, idx, bbox, class_id, instance_id, masks_t, cv_bgr, height, width, do_clip_frame):
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width - 1))
        y2 = max(0, min(int(y2), height - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        if masks_t is None or idx >= masks_t.shape[0]:
            return None

        m = masks_t[idx].detach().cpu().numpy()
        if m.shape[0] != height or m.shape[1] != width:
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        binary_mask = (m > 0.5).astype(np.uint8)

        M = cv2.moments(binary_mask)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

        entry = {
            "class_id": int(class_id),
            "instance_id": int(instance_id),
            "object_name": self.class_id_to_name(int(class_id)),
            "centroid": (cx, cy),
            "embedding": None,
            "crop": None,
        }

        if do_clip_frame:
            sx1, sy1, sx2, sy2 = CLIPProcessor.compute_square_crop(
                x1, y1, x2, y2, width, height, self.square_crop_scale
            )
            if sx2 > sx1 and sy2 > sy1:
                img_crop = cv_bgr[sy1:sy2, sx1:sx2]
                mask_crop = binary_mask[sy1:sy2, sx1:sx2]
                if img_crop.size > 0 and mask_crop.size > 0:
                    neutral_bg = np.full_like(img_crop, 122)
                    mask_u8 = (mask_crop * 255).astype(np.uint8)
                    bg_mask = cv2.bitwise_not(mask_u8)
                    fg = cv2.bitwise_and(img_crop, img_crop, mask=mask_u8)
                    bg = cv2.bitwise_and(neutral_bg, neutral_bg, mask=bg_mask)
                    final_crop = cv2.add(fg, bg)
                    entry["crop"] = final_crop

        return entry

    def log_detections(self, detections):
        for det in detections:
            emb = det.get("embedding")
            prob_goal = None
            if emb is not None and self.goal_text_embedding is not None:
                prob_goal = self.clip.compute_sigmoid_probs(emb, self.goal_text_embedding)

            if prob_goal is not None:
                self.get_logger().info(
                    f"ID {det['instance_id']} ({det['object_name']}): similarity={prob_goal}"
                )
            else:
                self.get_logger().info(
                    f"ID {det['instance_id']} ({det['object_name']}): similarity=unavailable"
                )

def main(args=None):
    rclpy.init(args=args)
    node = FlatVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
