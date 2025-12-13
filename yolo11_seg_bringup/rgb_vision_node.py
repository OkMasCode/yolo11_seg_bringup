#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import open_clip
from PIL import Image as PILImage
import json
import os

# Import your custom interface
from yolo11_seg_interfaces.msg import DetectedObject

import torch
import open_clip  # Changed from 'clip'
import cv2
import numpy as np
from PIL import Image as PILImage

class CLIPProcessor:
    """Handles SigLIP model loading and embedding generation."""
    
    def __init__(self, device='cpu', model_name='ViT-B-16-SigLIP', pretrained='webli'):
        """
        Initialize SigLIP processor.
        """
        self.device = device
        print(f"Loading SigLIP model: {model_name} ({pretrained})...")
        
        # 1. Load SigLIP Model & Transforms
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained, 
            device=device
        )
        
        # 2. Load SigLIP Tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.model.eval()

    def encode_text(self, text):
        """
        Encode text prompt(s) to a single normalized SigLIP embedding.
        Supports PROMPT ENSEMBLING.
        """
        if text is None: return None
        
        # Handle Input (String vs List)
        if isinstance(text, str):
            if not text.strip(): return None
            text = [text]
        elif isinstance(text, list):
            text = [t for t in text if isinstance(t, str) and t.strip()]
            if not text: return None

        with torch.no_grad():
            # Tokenize using OpenCLIP tokenizer
            tokens = self.tokenizer(text).to(self.device)
            
            # Encode
            feats = self.model.encode_text(tokens)
            
            # Normalize
            feats = feats / feats.norm(dim=-1, keepdim=True)
            
            # Ensemble (Average) & Re-Normalize
            ensemble_feat = feats.mean(dim=0, keepdim=True)
            ensemble_feat = ensemble_feat / ensemble_feat.norm(dim=-1, keepdim=True)
            
        return ensemble_feat.squeeze(0).detach().cpu().numpy()

    def apply_clahe(self, image_bgr):
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        # Convert to LAB and apply CLAHE
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        contrast_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Boost Saturation
        hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.5, 0, 255)
        hsv_boosted = cv2.merge([h, s, v]).astype(np.uint8)
        final = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
        
        return final

    def encode_images_batch(self, crops_bgr):
        """Encode a batch of BGR image crops to SigLIP embeddings."""
        if not crops_bgr:
            return []
        
        processed_crops = []
        for c in crops_bgr:
            # Apply CLAHE + Saturation Boost
            enhanced_c = self.apply_clahe(c) 
            
            # Convert BGR to RGB
            rgb_c = cv2.cvtColor(enhanced_c, cv2.COLOR_BGR2RGB)
            processed_crops.append(PILImage.fromarray(rgb_c))
        
        # Stack and Preprocess
        image_input = torch.stack([self.preprocess(img) for img in processed_crops]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()
    
    def get_sigmoid_probability(self, image_embedding, text_embedding):
        """
        Converts raw SigLIP dot product into a readable 0-1 probability.
        Formula: sigmoid( dot_product * scale + bias )
        """
        if image_embedding is None or text_embedding is None:
            return 0.0
            
        # Ensure numpy
        img = np.array(image_embedding)
        txt = np.array(text_embedding)
        
        # 1. Compute Raw Dot Product (Cosine Similarity)
        dot_product = np.dot(img, txt)
        
        # 2. Get Learned Scale & Bias from the Model
        with torch.no_grad():
            logit_scale = self.model.logit_scale.exp().item()
            logit_bias = self.model.logit_bias.item()
            
        # 3. Apply Sigmoid Transformation
        logits = (dot_product * logit_scale) + logit_bias
        probability = 1.0 / (1.0 + np.exp(-logits))
        
        return probability

    @staticmethod
    def compute_square_crop(x1, y1, x2, y2, width, height, scale=1.1):
        """Compute square crop coordinates."""
        bw, bh = x2 - x1, y2 - y1
        side = int(max(bw, bh) * scale)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        sx1 = max(0, int(cx - side / 2))
        sy1 = max(0, int(cy - side / 2))
        sx2 = min(width, int(cx + side / 2))
        sy2 = min(height, int(cy + side / 2))
        return sx1, sy1, sx2, sy2
class RGBVisionNode(Node):
    """
    Lightweight RGB-only YOLO+CLIP Node.
    - Runs YOLOv11 instance segmentation.
    - Computes 2D Centroids from masks.
    - Generates CLIP embeddings.
    - Publishes DetectedObject messages.
    """
    def __init__(self):
        super().__init__('rgb_vision_node')
        
        # --- Parameters ---
        self.declare_parameter('model_path', '/home/sensor/yolo11n-seg.engine')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('annotated_topic', '/yolo/annotated_image')
        self.declare_parameter('markers_topic', '/yolo/centroid_markers')
        self.declare_parameter('conf', 0.45)
        self.declare_parameter('clip_every_n_frames', 3) # Optimization: don't run CLIP every frame
        self.declare_parameter('robot_command_file', '/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json')
        self.declare_parameter('command_check_interval', 1.0)  # Check file every 1 second

        self.model_path = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.anno_topic = self.get_parameter('annotated_topic').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.conf_thres = self.get_parameter('conf').value
        self.clip_skip = self.get_parameter('clip_every_n_frames').value
        self.robot_command_file = self.get_parameter('robot_command_file').value
        self.command_check_interval = self.get_parameter('command_check_interval').value

        # --- Setup ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # Load Models
        self.get_logger().info(f"Loading YOLO: {self.model_path}")
        self.yolo = YOLO(self.model_path, task='segment')
        
        self.get_logger().info("Loading CLIP...")
        self.clip = CLIPProcessor(
            device=self.device, 
            model_name="ViT-B-16-SigLIP", 
            pretrained="webli"
        )
        # Track goal text and embedding
        self.current_clip_prompt = None
        self.goal_text_embedding = None
        self.distractor_embedding = None
        self.current_distractor_prompt = None

        # Initial load of robot command
        self._load_clip_prompt()

        # ROS Comm
        self.bridge = CvBridge()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.sub_img = self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.pub_det = self.create_publisher(DetectedObject, self.detections_topic, 10)
        self.pub_anno = self.create_publisher(Image, self.anno_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.markers_topic, 10)

        # Timer to periodically check robot_command.json for updates
        self.command_timer = self.create_timer(self.command_check_interval, self._load_clip_prompt)

        self.frame_count = 0
        
        # Define class names (COCO)
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
        
        self.get_logger().info("RGB Vision Node Ready.")

    def _load_clip_prompt(self):
        """Load clip_prompt from robot_command.json and update text embedding if changed."""
        try:
            if not os.path.exists(self.robot_command_file):
                return
            
            with open(self.robot_command_file, 'r') as f:
                data = json.load(f)
            
            # Get the prompt (can be str or list)
            clip_prompt = data.get('clip_prompt', None)
            if clip_prompt != self.current_clip_prompt:
                self.current_clip_prompt = clip_prompt
                if self._has_content(clip_prompt):
                    self.goal_text_embedding = self.clip.encode_text(clip_prompt)
                    self.get_logger().info(f"Updated Goal Ensemble: {clip_prompt}")
                else:
                    self.goal_text_embedding = None

            # 2. Update DISTRACTOR Prompt (NEW)
            distractor_prompt = data.get('distractor', None)
            if distractor_prompt != self.current_distractor_prompt:
                self.current_distractor_prompt = distractor_prompt
                if self._has_content(distractor_prompt):
                    # We encode the distractor list just like the goal list
                    self.distractor_embedding = self.clip.encode_text(distractor_prompt)
                    self.get_logger().info(f"Updated Distractor Ensemble: {distractor_prompt}")
                else:
                    self.distractor_embedding = None

        except Exception as e:
            self.get_logger().error(f"Error loading robot_command.json: {e}")

    def _has_content(self, text_obj):
        if isinstance(text_obj, str) and text_obj.strip(): return True
        if isinstance(text_obj, list) and len(text_obj) > 0: return True
        return False

    def class_id_to_name(self, class_id: int) -> str:
        """Convert class ID to class name."""
        if 0 <= class_id < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id]
        return f"class_{class_id}"

    def image_callback(self, msg: Image):
        self.frame_count += 1
        try:
            # 1. Convert Image
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = img.shape[:2]

            # 2. YOLO Inference
            results = self.yolo.track(img, persist=True, conf=self.conf_thres, verbose=False, retina_masks=True)
            res = results[0]

            if not res.boxes:
                return

            # Prepare data
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            classes = res.boxes.cls.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.zeros(len(boxes), dtype=int)
            masks = res.masks.data if res.masks is not None else None # (N, H, W) tensor on device

            detections_to_process = []
            crops_for_clip = []

            for i, box in enumerate(boxes):
                if classes[i] == 0: continue # Skip person if needed
                
                class_name = self.class_id_to_name(classes[i])
                x1, y1, x2, y2 = box
                
                # --- Centroid & Mask Logic ---
                cx, cy = float((x1+x2)/2), float((y1+y2)/2)
                binary_mask = None
                
                if masks is not None:
                    m = masks[i].cpu().numpy()
                    # Ensure mask size matches image
                    if m.shape[0] != h or m.shape[1] != w:
                         m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    binary_mask = (m > 0.5).astype(np.uint8) * 255
                    
                    # Compute Refined Centroid
                    M = cv2.moments(binary_mask)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                else:
                    # Fallback mask (full white)
                    binary_mask = np.ones((h, w), dtype=np.uint8) * 255

                # --- Extract Masked Crop (Gray Background) ---
                sx1, sy1, sx2, sy2 = CLIPProcessor.compute_square_crop(x1, y1, x2, y2, w, h)
                img_crop = img[sy1:sy2, sx1:sx2]
                mask_crop = binary_mask[sy1:sy2, sx1:sx2]
                
                if img_crop.size == 0 or mask_crop.size == 0: continue

                # Apply Gray Masking (Critical for Distractor logic)
                neutral_bg = np.full_like(img_crop, 122) # 122 is neutral gray
                bg_mask = cv2.bitwise_not(mask_crop)
                fg = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
                bg = cv2.bitwise_and(neutral_bg, neutral_bg, mask=bg_mask)
                final_crop = cv2.add(fg, bg)

                detections_to_process.append({
                    "id": ids[i],
                    "class": class_name,
                    "centroid": (cx, cy),
                })
                crops_for_clip.append(final_crop)

            # --- Batch SigLIP Inference ---
            embeddings = []
            if self.frame_count % self.clip_skip == 0 and crops_for_clip:
                embeddings = self.clip.encode_images_batch(crops_for_clip)

            # --- Publish ---
            marker_array = MarkerArray()
            for i, det in enumerate(detections_to_process):
                msg_det = DetectedObject()
                msg_det.timestamp = msg.header.stamp
                msg_det.object_id = int(det["id"])
                msg_det.object_name = det["class"]
                msg_det.centroid = Vector3(x=float(det["centroid"][0]), y=float(det["centroid"][1]), z=0.0)
                
                if i < len(embeddings):
                    current_emb = embeddings[i]
                    msg_det.embedding = current_emb.tolist()
                    
                    # --- NEW: DISTRACTOR COMPARISON LOGIC ---
                    if self.goal_text_embedding is not None:
                        # 1. Calculate Goal Probability
                        prob_goal = self.clip.get_sigmoid_probability(current_emb, self.goal_text_embedding)
                        msg_det.similarity = float(prob_goal) # If your msg definition has this field

                        # 2. Calculate Distractor Probability (if exists)
                        prob_distractor = 0.0
                        if self.distractor_embedding is not None:
                            prob_distractor = self.clip.get_sigmoid_probability(current_emb, self.distractor_embedding)

                        # 3. Log the decision data (You can verify this in terminal)
                        self.get_logger().info(
                            f"ID {det['id']} ({det['class']}): Goal={prob_goal:.1%} | Distractor={prob_distractor:.1%}"
                        )
                        
                        # Note: The logic "If Goal > Distractor + 0.1" can be implemented here 
                        # to set a 'is_target' flag if your msg supports it, or filtered downstream.
                
                self.pub_det.publish(msg_det)

                # Create marker for centroid visualization
                marker = Marker()
                marker.header.frame_id = msg.header.frame_id if msg.header.frame_id else "camera_frame"
                marker.header.stamp = msg.header.stamp
                marker.ns = "centroids"
                marker.id = int(det["id"])
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # Position (converting pixel coords to camera frame - for 2D overlay use z=0)
                marker.pose.position.x = det["centroid"][0] / 1000.0  # Scale for visualization
                marker.pose.position.y = det["centroid"][1] / 1000.0
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Scale (size of marker)
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                
                # Color (green with transparency)
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.8
                
                marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                marker_array.markers.append(marker)

                # Create text marker for class label
                text_marker = Marker()
                text_marker.header.frame_id = msg.header.frame_id if msg.header.frame_id else "camera_frame"
                text_marker.header.stamp = msg.header.stamp
                text_marker.ns = "class_labels"
                text_marker.id = int(det["id"])
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                # Position slightly offset from centroid
                text_marker.pose.position.x = det["centroid"][0] / 1000.0
                text_marker.pose.position.y = det["centroid"][1] / 1000.0
                text_marker.pose.position.z = 0.05  # Slightly above
                text_marker.pose.orientation.w = 1.0
                
                # Text content
                text_marker.text = det["class"]
                
                # Scale (text height)
                text_marker.scale.z = 0.05
                
                # Color (white)
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                
                text_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                marker_array.markers.append(text_marker)

            # Publish MarkerArray
            if marker_array.markers:
                self.pub_markers.publish(marker_array)

            # 6. Publish Annotation
            if self.pub_anno.get_subscription_count() > 0:
                annotated_frame = res.plot()
                self.pub_anno.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))

        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RGBVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()