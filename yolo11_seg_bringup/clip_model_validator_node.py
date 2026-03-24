import threading
import json
import os
from time import perf_counter

import cv2
import numpy as np
import open_clip
import rclpy
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class ClipModelValidatorNode(Node):
    """
    Validator node that runs YOLO segmentation and computes prompt similarity
    for every detected object using CLIP/SigLIP embeddings.
    """

    def __init__(self):
        super().__init__("clip_model_validator_node")

        # --- Parameters ---
        self.declare_parameter("image_topic", "/camera/rgb")
        self.declare_parameter("similarity_topic", "/clip/match_score")
        self.declare_parameter("annotated_image_topic", "/clip/annotated_image")
        self.declare_parameter("clip_model_name", "ViT-B-16-SigLIP")
        self.declare_parameter("clip_pretrained", "webli")
        self.declare_parameter(
            "prompt_file",
            "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/clip_prompt.json",
        )
        self.declare_parameter("prompt_check_interval", 30.0)
        self.declare_parameter("yolo_model_path", "/workspaces/yoloe-26l-seg.pt")
        self.declare_parameter("yolo_imgsz", 640)
        self.declare_parameter("yolo_conf", 0.45)
        self.declare_parameter("yolo_iou", 0.35)
        self.declare_parameter("square_crop_scale", 1.2)
        self.declare_parameter("masked_score_weight", 0.5)
        self.declare_parameter("unmasked_score_weight", 0.5)

        self.image_topic = self.get_parameter("image_topic").value
        self.similarity_topic = self.get_parameter("similarity_topic").value
        self.annotated_image_topic = self.get_parameter("annotated_image_topic").value
        self.clip_model_name = self.get_parameter("clip_model_name").value
        self.clip_pretrained = self.get_parameter("clip_pretrained").value
        self.prompt_file = self.get_parameter("prompt_file").value
        self.prompt_check_interval = float(self.get_parameter("prompt_check_interval").value)
        self.yolo_model_path = self.get_parameter("yolo_model_path").value
        self.yolo_imgsz = int(self.get_parameter("yolo_imgsz").value)
        self.yolo_conf = float(self.get_parameter("yolo_conf").value)
        self.yolo_iou = float(self.get_parameter("yolo_iou").value)
        self.square_crop_scale = float(self.get_parameter("square_crop_scale").value)
        self.masked_score_weight = float(self.get_parameter("masked_score_weight").value)
        self.unmasked_score_weight = float(self.get_parameter("unmasked_score_weight").value)

        # --- Device & Model Setup ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading model: {self.clip_model_name} on {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.clip_model_name,
            pretrained=self.clip_pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        self.model.eval()

        # Cache constants to avoid GPU-CPU sync in every similarity call.
        with torch.no_grad():
            self.cached_logit_scale = (
                float(self.model.logit_scale.exp().item())
                if hasattr(self.model, "logit_scale")
                else 1.0
            )
            self.cached_logit_bias = (
                float(self.model.logit_bias.item())
                if hasattr(self.model, "logit_bias")
                else 0.0
            )

        self.CLASS_NAMES = ["bottle"]
        
        self.get_logger().info(f"Loading YOLO segmentation model: {self.yolo_model_path}")
        self.yolo = YOLO(self.yolo_model_path, task="segment")
        self.yolo.set_classes(self.CLASS_NAMES)

        # --- Concurrency & State Setup ---
        self.bridge = CvBridge()
        self.state_lock = threading.Lock()

        self.current_prompt = None
        self.current_text_embedding = None

        self.avg_window = 30
        self.avg_score_sum = 0.0
        self.avg_score_count = 0

        self.timing_stats = {
            "text_encoding": [],
            "cv_bridge": [],
            "yolo_inference": [],
            "image_preprocess": [],
            "image_inference": [],
            "match_score_compute": [],
            "total_frame": [],
        }

        # --- ROS2 Communications (Multi-Threaded) ---
        self.image_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

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
            callback_group=self.image_cb_group,
        )
        self.sim_pub = self.create_publisher(Float32, self.similarity_topic, 10)
        self.annotated_image_pub = self.create_publisher(
            Image,
            self.annotated_image_topic,
            10,
        )

        # Start timer for prompt updates.
        self._load_prompt_from_json()
        self.prompt_timer = self.create_timer(
            self.prompt_check_interval,
            self._load_prompt_from_json,
            callback_group=self.timer_cb_group,
        )

        self.get_logger().info("ClipModelValidatorNode ready.")

    @staticmethod
    def _pad_to_square(image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        h, w = image_bgr.shape[:2]
        if h == w:
            return image_bgr

        side = max(h, w)
        canvas = np.full((side, side, 3), 128, dtype=np.uint8)

        x_offset, y_offset = (side - w) // 2, (side - h) // 2
        canvas[y_offset : y_offset + h, x_offset : x_offset + w] = image_bgr
        return canvas

    @staticmethod
    def _compute_square_crop(x1, y1, x2, y2, width, height, scale=1.2):
        bw, bh = x2 - x1, y2 - y1
        side = int(max(bw, bh) * scale)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        sx1 = max(0, int(cx - side / 2))
        sy1 = max(0, int(cy - side / 2))
        sx2 = min(width, int(cx + side / 2))
        sy2 = min(height, int(cy + side / 2))
        return sx1, sy1, sx2, sy2

    def _build_prompt_list(self, label: str) -> list:
        label = label.strip()
        return [
            f"a photo of a {label}.",
            f"a bad photo of a {label}.",
            f"a photo of the large {label}.",
            f"a photo of the small {label}.",
            f"a cropped photo of a {label}.",
            f"a close-up photo of a {label}.",
            f"a clear photo of a {label}.",
        ]

    def _encode_text(self, text_list: list):
        if not text_list:
            return None

        text_start = perf_counter()
        with torch.no_grad():
            try:
                tokens = self.tokenizer(text_list).to(self.device)
            except AttributeError:
                if hasattr(self.tokenizer, "tokenizer"):
                    hf_tokenizer = self.tokenizer.tokenizer
                    context_length = getattr(self.tokenizer, "context_length", 77)
                    encoded = hf_tokenizer(
                        text_list,
                        padding="max_length",
                        truncation=True,
                        max_length=context_length,
                        return_tensors="pt",
                    )
                    tokens = encoded["input_ids"].to(self.device)
                else:
                    raise

            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

            ensemble_feature = features.mean(dim=0, keepdim=True)
            ensemble_feature = ensemble_feature / ensemble_feature.norm(
                dim=-1,
                keepdim=True,
            )

        self.timing_stats["text_encoding"].append((perf_counter() - text_start) * 1000.0)
        return ensemble_feature.squeeze(0).detach().cpu().numpy()

    def _encode_images_batch(self, images_bgr: list):
        if not images_bgr:
            return []

        preprocess_start = perf_counter()
        image_tensors = []
        for image_bgr in images_bgr:
            if image_bgr is None or image_bgr.size == 0:
                continue
            squared_bgr = self._pad_to_square(image_bgr)
            image_rgb = cv2.cvtColor(squared_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            image_tensors.append(self.preprocess(pil_image))

        if not image_tensors:
            return []

        self.timing_stats["image_preprocess"].append((perf_counter() - preprocess_start) * 1000.0)

        with torch.no_grad():
            inference_start = perf_counter()
            image_input = torch.stack(image_tensors, dim=0).to(self.device)
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)

        self.timing_stats["image_inference"].append((perf_counter() - inference_start) * 1000.0)
        return [embedding for embedding in features.detach().cpu().numpy()]

    def _prepare_crops(self, cv_bgr, mask_uint8, bbox_xyxy):
        """
        Build square crop pair:
        1) masked crop with neutral-gray background outside object mask
        2) unmasked crop directly from the original image
        """
        height, width = cv_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy

        sx1, sy1, sx2, sy2 = self._compute_square_crop(
            x1,
            y1,
            x2,
            y2,
            width,
            height,
            self.square_crop_scale,
        )
        if sx2 <= sx1 or sy2 <= sy1:
            return None, None

        img_crop = cv_bgr[sy1:sy2, sx1:sx2]
        unmasked_crop = img_crop.copy()
        mask_crop = mask_uint8[sy1:sy2, sx1:sx2]

        if img_crop.size == 0 or mask_crop.size == 0:
            return None, None

        mask_crop = cv2.dilate(mask_crop, np.ones((5, 5), np.uint8), iterations=1)
        neutral_bg = np.full_like(img_crop, 122)

        bg_mask = cv2.bitwise_not(mask_crop)
        fg = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
        bg = cv2.bitwise_and(neutral_bg, neutral_bg, mask=bg_mask)
        masked_crop = cv2.add(fg, bg)
        return masked_crop, unmasked_crop

    def _load_prompt_from_json(self):
        prompt_path = self.prompt_file
        fallback_prompt_path = "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/clip_prompt.json"

        if not os.path.exists(prompt_path):
            if prompt_path != fallback_prompt_path and os.path.exists(fallback_prompt_path):
                self.get_logger().warn(
                    f"Prompt file not found at '{prompt_path}'. Using fallback '{fallback_prompt_path}'.",
                    throttle_duration_sec=10.0,
                )
                prompt_path = fallback_prompt_path
            else:
                self.get_logger().warn(
                    f"Prompt file not found: '{prompt_path}'. CLIP scoring is paused until this file exists.",
                    throttle_duration_sec=10.0,
                )
                return

        try:
            with open(prompt_path, "r") as file_handle:
                data = json.load(file_handle)

            prompt = data.get("clip_prompt", None)
            if not isinstance(prompt, str) or not prompt.strip():
                return
            prompt = prompt.strip()

            if prompt == self.current_prompt:
                return

            prompt_list = self._build_prompt_list(prompt)
            text_embedding = self._encode_text(prompt_list)
            if text_embedding is None:
                return

            with self.state_lock:
                self.current_prompt = prompt
                self.current_text_embedding = text_embedding

            self.get_logger().info(
                f"Updated and ensembled prompt from '{prompt_path}': '{prompt}'"
            )

        except Exception as exc:
            self.get_logger().error(f"Failed reading prompt file: {exc}")

    def image_callback(self, msg: Image):
        with self.state_lock:
            text_embedding = (
                None
                if self.current_text_embedding is None
                else self.current_text_embedding.copy()
            )

        try:
            frame_start = perf_counter()

            cv_bridge_start = perf_counter()
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.timing_stats["cv_bridge"].append((perf_counter() - cv_bridge_start) * 1000.0)

            yolo_start = perf_counter()
            results = self.yolo.track(
                source=cv_bgr,
                imgsz=self.yolo_imgsz,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                retina_masks=True,
                stream=False,
                verbose=False,
                persist=True,
                tracker="botsort.yaml",
            )
            self.timing_stats["yolo_inference"].append((perf_counter() - yolo_start) * 1000.0)

            if not results:
                self.get_logger().warn("YOLO returned no result entries", throttle_duration_sec=2.0)
                return

            res = results[0]

            # Publish YOLO-rendered image for visualization/debugging.
            annotated_bgr = res.plot()
            if annotated_bgr is not None:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding="bgr8")
                annotated_msg.header = msg.header
                self.annotated_image_pub.publish(annotated_msg)
                self.get_logger().info(
                    f"Published annotated image on {self.annotated_image_topic}",
                    throttle_duration_sec=2.0,
                )

            if not hasattr(res, "boxes") or res.boxes is None or len(res.boxes) == 0:
                self.get_logger().info("No detections in current frame", throttle_duration_sec=2.0)
                return

            if text_embedding is None:
                self.get_logger().warn(
                    "Prompt embedding not ready; published annotated image but skipped CLIP scoring",
                    throttle_duration_sec=2.0,
                )
                return

            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            clss = res.boxes.cls.detach().cpu().numpy().astype(int)
            confs = (
                res.boxes.conf.detach().cpu().numpy()
                if res.boxes.conf is not None
                else np.zeros(len(clss), dtype=np.float32)
            )
            ids = (
                res.boxes.id.detach().cpu().numpy().astype(int)
                if res.boxes.id is not None
                else np.arange(len(clss), dtype=np.int32)
            )

            names = [res.names.get(int(c), str(int(c))) for c in clss]
            h, w = cv_bgr.shape[:2]
            masks_np = None
            if hasattr(res, "masks") and res.masks is not None and res.masks.data is not None:
                masks_np = res.masks.data.detach().cpu().numpy()

            detection_meta = []
            masked_crops = []
            unmasked_crops = []
            for idx in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[idx]
                x1 = max(0, min(int(x1), w - 1))
                y1 = max(0, min(int(y1), h - 1))
                x2 = max(0, min(int(x2), w - 1))
                y2 = max(0, min(int(y2), h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                if masks_np is not None and idx < masks_np.shape[0]:
                    instance_mask = masks_np[idx]
                    if instance_mask.shape[0] != h or instance_mask.shape[1] != w:
                        instance_mask = cv2.resize(
                            instance_mask,
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask_uint8 = (instance_mask > 0.5).astype(np.uint8) * 255
                else:
                    mask_uint8 = np.zeros((h, w), dtype=np.uint8)
                    mask_uint8[y1:y2, x1:x2] = 255

                masked_crop, unmasked_crop = self._prepare_crops(
                    cv_bgr,
                    mask_uint8,
                    (x1, y1, x2, y2),
                )
                if masked_crop is None or unmasked_crop is None:
                    continue

                detection_meta.append(
                    {
                        "instance_id": int(ids[idx]),
                        "class_name": str(names[idx]),
                        "confidence": float(confs[idx]),
                    }
                )
                masked_crops.append(masked_crop)
                unmasked_crops.append(unmasked_crop)

            if not masked_crops:
                self.get_logger().info(
                    "Detections found but no valid masked crops",
                    throttle_duration_sec=2.0,
                )
                return

            masked_embeddings = self._encode_images_batch(masked_crops)
            unmasked_embeddings = self._encode_images_batch(unmasked_crops)
            if not masked_embeddings or not unmasked_embeddings:
                return

            pair_count = min(len(masked_embeddings), len(unmasked_embeddings), len(detection_meta))
            if pair_count == 0:
                return

            score_start = perf_counter()
            best_score = 0.0
            total_score = 0.0

            for idx in range(pair_count):
                masked_score = self._compute_match_score(masked_embeddings[idx], text_embedding)
                unmasked_score = self._compute_match_score(unmasked_embeddings[idx], text_embedding)
                match_score = (
                    (self.masked_score_weight * masked_score)
                    + (self.unmasked_score_weight * unmasked_score)
                )
                meta = detection_meta[idx]

                self.get_logger().info(
                    "Object similarity: "
                    f"id={meta['instance_id']} "
                    f"class='{meta['class_name']}' "
                    f"conf={meta['confidence']:.3f} "
                    f"score={match_score:.2f} "
                    f"(masked={masked_score:.2f}, unmasked={unmasked_score:.2f})"
                )

                total_score += match_score
                if match_score > best_score:
                    best_score = match_score

            self.timing_stats["match_score_compute"].append((perf_counter() - score_start) * 1000.0)

            # Preserve scalar output for downstream consumers.
            sim_msg = Float32()
            sim_msg.data = float(best_score)
            self.sim_pub.publish(sim_msg)

            self.timing_stats["total_frame"].append((perf_counter() - frame_start) * 1000.0)

            self.avg_score_sum += total_score / pair_count
            self.avg_score_count += 1
            if self.avg_score_count >= self.avg_window:
                self._log_and_reset_timing_stats()
                self.avg_score_sum = 0.0
                self.avg_score_count = 0

        except Exception as exc:
            self.get_logger().error(f"Failed to process image/prompt pair: {exc}")

    @staticmethod
    def _safe_sigmoid(value: float) -> float:
        clipped = np.clip(value, -60.0, 60.0)
        return float(1.0 / (1.0 + np.exp(-clipped)))

    def _compute_match_score(self, image_embedding, text_embedding) -> float:
        image_vec = np.asarray(image_embedding, dtype=np.float32).flatten()
        text_vec = np.asarray(text_embedding, dtype=np.float32).flatten()

        dot_product = float(np.dot(image_vec, text_vec))
        logits = (dot_product * self.cached_logit_scale) + self.cached_logit_bias
        return self._safe_sigmoid(logits) * 100.0

    def _log_and_reset_timing_stats(self):
        total_times = self.timing_stats.get("total_frame", [])
        if total_times and self.avg_score_count > 0:
            avg_total = float(np.mean(np.asarray(total_times, dtype=np.float32)))
            avg_match_score = self.avg_score_sum / self.avg_score_count
            self.get_logger().info(
                f"Avg ({len(total_times)} frames): total_time={avg_total:.2f}ms, match_score={avg_match_score:.2f}"
            )

        for key in self.timing_stats:
            self.timing_stats[key] = []


def main(args=None):
    rclpy.init(args=args)
    node = ClipModelValidatorNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()