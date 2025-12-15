import os
import rclpy
from rclpy.node import Node
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField, Image, CameraInfo, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

import torch
import cv2
import numpy as np
import struct
from cv_bridge import CvBridge
import threading
import json

import open_clip
from PIL import Image as PILImage
from ultralytics import YOLO

from yolo11_seg_interfaces.msg import DetectedObject

# ------------------ UTILITIES ----------------- #
class CLIPProcessor:
    """
    Handles SigLIP model processing tasks:
    - Square image cropping
    - Image preprocessing (CLAHE, saturation adjustment)
    - Ecoding text and images for model input
    - Computing sigmoid probabilities
    """
    def __init__(self, device='cpu', model_name='ViT-B-16-SigLIP', pretrained='webli'):
        """
        Iniialize CLIPProcessor
        """
        self.device = device
        print(f"Loading SigLIP model: {model_name} ({pretrained})...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=device
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()

    @staticmethod
    def compute_square_crop(x1,y1,x2,y2, width, height, scale = 1.2):
        """
        Compute square crop coordinates
        """
        bw, bh = x2 - x1, y2 - y1
        side = int(max(bw, bh) * scale)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        sx1 = max(0, int(cx - side / 2))
        sy1 = max(0, int(cy - side / 2))
        sx2 = min(width, int(cx + side / 2))
        sy2 = min(height, int(cy + side / 2))
        return sx1, sy1, sx2, sy2
        
    def encode_text(self, text):
        """
        Encode text prompts to a single normalized SigLIP embedding.
        Supports prompt ensembling by averaging multiple prompts.
        """
        if text is None: return None

        if isinstance(text, str):
            if not text.strip():
                return None
            text = [text]
        elif isinstance(text, list):
            text = [t for t in text if isinstance(t, str) and t.strip()]
            if not text:  
                return None
        
        with torch.no_grad():
            tokens = self.tokenizer(text).to(self.device)

            feats = self.model.encode_text(tokens)

            feats = feats / feats.norm(dim=-1, keepdim=True)

            ensamble_feat = feats.mean(dim=0, keepdim=True)
            ensamble_feat = ensamble_feat / ensamble_feat.norm(dim=-1, keepdim=True)

        return ensamble_feat.squeeze(0).detach().cpu().numpy()

    def preproc_image(self, image_bgr):
        """
        Applies Contrast Limited Adaptive Histogram Equaization (CLAHE)
        and saturation adjustment.
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        contrast_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.25, 0, 255)
        hsv_boosted = cv2.merge([h, s, v]).astype(np.uint8)

        final = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
        return final

    def encode_images(self, crops_bgr):
        """
        Encode a batch of image crops to SigLIP embeddings.
        """
        if not crops_bgr: return []

        processed_crops = []
        for c in crops_bgr:
            enhanced_c = self.preproc_image(c)
            rgb_c = cv2.cvtColor(enhanced_c, cv2.COLOR_BGR2RGB)
            processed_crops.append(PILImage.fromarray(rgb_c))

        image_input = torch.stack([self.preprocess(img) for img in processed_crops]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy()
    
    def compute_sigmoid_probs(self, image_embedding, text_embedding):
        """
        Converts raw SigLIP dot product into a readable probability.
        Formula: sigmoid( dot_product * scale + bias)
        """
        if image_embedding is None or text_embedding is None:
            return None

        img = np.array(image_embedding)
        txt = np.array(text_embedding)
        
        dot_product = np.dot(img, txt)
        
        with torch.no_grad():
            logit_scale = self.model.logit_scale.exp().item()
            logit_bias = self.model.logit_bias.item()
        
        logits = (dot_product * logit_scale) + logit_bias
        probs = 1 / (1 + np.exp(-logits))
        return probs
class PointCloudProcessor:
    """ 
    Handles PointCloud generation from depth and segmentation masks. 
    """
    def __init__(self, fx, fy, cx, cy, device, depth_scale = 1000.0, 
                 pc_downsample = 2, pc_max_range = 8.0, mask_threshold = 0.5):
         
        self.depth_scale = depth_scale
        self.pc_downsample = pc_downsample
        self.pc_max_range = pc_max_range
        self.mask_threshold = mask_threshold
        self.device = device

        self.fx_t = torch.tensor(fx, dtype=torch.float32, device=device)
        self.fy_t = torch.tensor(fy, dtype=torch.float32, device=device)
        self.cx_t = torch.tensor(cx, dtype=torch.float32, device=device)
        self.cy_t = torch.tensor(cy, dtype=torch.float32, device=device)

    @staticmethod
    def pack_rgb(r: int, g: int, b: int) -> float:
        """ 
        Pack 3x uint8 RGB into float32 for PointCloud2 'rgb' field. 
        """
        rgb_uint32 = (r << 16) | (g << 8) | b
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]
    
    @staticmethod
    def publish_pointcloud(points: np.ndarray, header, publisher):
        """
        Publish PointCloud2 message from points numpy array.
        """
        try:
            if points is None or points.size == 0:
                return

            dtype = np.dtype([
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.float32),
                ('class_id', np.int32),
                ('instance_id', np.int32),
            ])

            structured_points = np.zeros(points.shape[0], dtype=dtype)
            structured_points['x'] = points[:, 0].astype(np.float32)
            structured_points['y'] = points[:, 1].astype(np.float32)
            structured_points['z'] = points[:, 2].astype(np.float32)
            structured_points['rgb'] = points[:, 3].astype(np.float32)
            structured_points['class_id'] = points[:, 4].astype(np.int32)
            structured_points['instance_id'] = points[:, 5].astype(np.int32)

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='class_id', offset=16, datatype=PointField.INT32, count=1),
                PointField(name='instance_id', offset=20, datatype=PointField.INT32, count=1),
            ]
            cloud_msg = point_cloud2.create_cloud(header, fields, structured_points)
            publisher.publish(cloud_msg)
        except Exception as e:
            print(f"Error publishing pointcloud: {e}")

    def process_detection(self, mask_t, depth_t, valid_mask_t, class_id, instance_id, 
                         rgb_color, scale_factor, min_points=10):
        """ 
        Process a single detection to generate pointcloud segment. 
        """
        obj_mask_t = (mask_t >= self.mask_threshold).to(self.device)
        valid_t = valid_mask_t & obj_mask_t

        v_coords_t, u_coords_t = valid_t.nonzero(as_tuple=True)
        if v_coords_t.shape[0] < min_points:
            return None, None # Not enough points
        
        z_vals_t = (depth_t[v_coords_t, u_coords_t].mul_(scale_factor))

        z_min = torch.quantile(z_vals_t, 0.1)
        z_max = torch.quantile(z_vals_t, 0.9)
        keep_mask_t = (z_vals_t >= z_min) & (z_vals_t <= z_max)
        if not torch.any(keep_mask_t):
            return None, None
        
        z_clean_t = z_vals_t[keep_mask_t]
        u_clean_t = u_coords_t[keep_mask_t].float()
        v_clean_t = v_coords_t[keep_mask_t].float()

        if self.pc_downsample and self.pc_downsample > 1:
            step_t = int(self.pc_downsample)
            idx = torch.arange(0, z_clean_t.shape[0], step_t, device=self.device, dtype=torch.long)
            z_clean_t = z_clean_t[idx]
            u_clean_t = u_clean_t[idx]
            v_clean_t = v_clean_t[idx]
        
        x_t = (u_clean_t - self.cx_t) * z_clean_t / self.fx_t
        y_t = (v_clean_t - self.cy_t) * z_clean_t / self.fy_t

        centroid = (
            float(torch.mean(x_t).item()),
            float(torch.mean(y_t).item()),
            float(torch.mean(z_clean_t).item())
        )

        N = x_t.shape[0]
        if N == 0:
            return None, None
        
        r, g, b = rgb_color
        rgb_packed = self.pack_rgb(r, g, b)

        instance_cloud_t = torch.stack(
            [
                x_t, 
                y_t, 
                z_clean_t,
                torch.full((N,), float(rgb_packed), dtype=torch.float32, device=self.device),
                torch.full((N,), int(class_id), dtype=torch.int32, device=self.device),
                torch.full((N,), int(instance_id), dtype=torch.int32, device=self.device),
            ],
            dim=1
        )
        return instance_cloud_t, centroid

    def prepare_depth_tensor(self, depth_img, encoding, scale_factor):
        """ 
        Convert depth image to GPU tensor with validity mask. 
        """
        depth_t = torch.from_numpy(depth_img.astype(np.float32)).to(self.device)
        valid_mask_t = (depth_t > 0) & (~torch.isnan(depth_t))
        if self.pc_max_range > 0.0:
            valid_mask_t = valid_mask_t & (depth_t * scale_factor <= self.pc_max_range)
        return depth_t, valid_mask_t
    
# -------------------- CLASS ------------------- #

class VisionNode(Node):
    """
    ROS2 Node that handles Vision computation
    """

    # ------------- Initialization ------------- #

    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info("VisionNode initialized\n")

        # ============= Parameters ============= #

        # Comunication parameters
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('enable_visualization', True)

        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.enable_visualization = bool(self.get_parameter('enable_visualization').value)

        # yolo parameters
        self.declare_parameter('model_path', '/home/sensor/yolo11n-seg.engine')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf', 0.45)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('retina_masks', True)

        self.model_path = self.get_parameter('model_path').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        self.retina_masks = bool(self.get_parameter('retina_masks').value)

        # CLIP parameters
        self.declare_parameter('CLIP_model_name', 'ViT-B-16-SigLIP')
        self.declare_parameter('robot_command_file', '/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json')
        self.declare_parameter('square_crop_scale', 1.2)

        self.CLIP_model_name = self.get_parameter('CLIP_model_name').value
        self.robot_command_file = self.get_parameter('robot_command_file').value
        self.square_crop_scale = float(self.get_parameter('square_crop_scale').value)

        # =========== Initialization =========== #

        self.pc_topic = '/vision/pointcloud'
        self.anno_topic = '/vision/annotated_image'
        self.markers_topic = '/vision/centroid_markers'
        self.detection_topic = '/vision/detections'

        self.depth_scale = 1000.0
        self.pc_downsample = 2
        self.pc_max_range = 8.0
        self.mask_threshold = 0.5

        self.frame_skip = 3
        self.prompt_check_interval = 10
        self.text_prompt = 'a photo of a chair'

        # Load YOLO model
        self.get_logger().info(f"Loading YOLO: {self.model_path}")
        self.yolo = YOLO(self.model_path, task='segment')    

        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on device: {self.device}\n")
        self.clip = CLIPProcessor(
            device=self.device, 
            model_name="ViT-B-16-SigLIP", 
            pretrained="webli"
        )

        self._load_clip_prompt()

        self.bridge = CvBridge()
        self.pc_processor = None

        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, self.image_topic, self.rgb_callback, qos_profile=qos_sensor)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile=qos_sensor)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_profile=qos_sensor)

        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, self.pc_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.detections_pub = self.create_publisher(DetectedObject, self.detection_topic, 10)
        if self.enable_visualization:
            self.vis_pub = self.create_publisher(Image, self.anno_topic, 10)


        self.command_timer = self.create_timer(self.prompt_check_interval, self._load_clip_prompt)

        self.frame_count = 0
        self.class_colors = {}
        self.current_clip_prompt = None
        self.goal_text_embedding = None
        self.distractor_embedding = None
        self.current_distractor_prompt = None
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth_msg = None
        self.warned_missing_intrinsics = False
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

        self.get_logger().info("RGB Vision Node Ready.")

    # ---------------- Callbacks --------------- #



    # --------------- Main Methods ------------- #

    def _load_clip_prompt(self):
        """
        Load clip_prompt and distractor from robot_command.json and update text embedding if changed.
        """
        try:
            if not os.path.exists(self.robot_command_file):
                return
            
            with open(self.robot_command_file, 'r') as f:
                data = json.load(f)
            
            # Update GOAL Prompt
            clip_prompt = data.get('clip_prompt', None)
            if clip_prompt != self.current_clip_prompt:
                self.current_clip_prompt = clip_prompt
                self.goal_text_embedding = self.clip.encode_text(clip_prompt)
                self.get_logger().info(f"Updated Goal Ensemble: {clip_prompt}")

            # Update DISTRACTOR Prompt
            distractor_prompt = data.get('distractor', None)
            if distractor_prompt != self.current_distractor_prompt:
                self.current_distractor_prompt = distractor_prompt
                self.distractor_embedding = self.clip.encode_text(distractor_prompt)
                self.get_logger().info(f"Updated Distractor Ensemble: {distractor_prompt}")

        except Exception as e:
            self.get_logger().error(f"Error loading robot_command.json: {e}")

    # ----------- Secondary Methods ------------ #

    def class_id_to_name(self, class_id: int) -> str:
        """
        Convert class ID to class name.
        """
        if 0 <= class_id < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id]
        return f"class_{class_id}"

    @staticmethod
    def get_color_for_class(class_id: str, class_colors: dict):
        """ 
        Deterministically generate a color for a given class ID. 
        """
        if class_id not in class_colors:
            h = abs(hash(class_id))
            r = (h >> 0) & 0xFF
            g = (h >> 8) & 0xFF
            b = (h >> 16) & 0xFF
            if r < 30 and g < 30 and b < 30:
                r = (r + 128) & 0xFF
                g = (g + 64) & 0xFF
            class_colors[class_id] = (r, g, b)
        return class_colors[class_id]