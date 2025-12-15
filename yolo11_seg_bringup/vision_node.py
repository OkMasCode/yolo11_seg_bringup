import os
import rclpy
from rclpy.node import Node
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField, Image, CameraInfo, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Vector3
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
        s = np.clip(s * 1.25, 0, 255).astype(np.uint8)
        hsv_boosted = cv2.merge([h, s, v])

        final = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
        return final

    def encode_images_batch(self, crops_bgr):
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
                 pc_downsample = 2, pc_max_range = 8.0):
         
        self.depth_scale = depth_scale
        self.pc_downsample = pc_downsample
        self.pc_max_range = pc_max_range
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
        obj_mask_t = (mask_t).to(self.device)
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
        self.enable_vis = bool(self.get_parameter('enable_visualization').value)

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

        self.frame_skip = 1
        self.prompt_check_interval = 10
        self.text_prompt = 'a photo of a chair'

        # Load YOLO model
        self.get_logger().info(f"Loading YOLO: {self.model_path}")
        self.model = YOLO(self.model_path, task='segment')    

        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP model on device: {self.device}\n")
        self.clip = CLIPProcessor(
            device=self.device, 
            model_name="ViT-B-16-SigLIP", 
            pretrained="webli"
        )

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
        if self.enable_vis:
            self.vis_pub = self.create_publisher(Image, self.anno_topic, 10)

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

        self._load_clip_prompt()
        self.command_timer = self.create_timer(self.prompt_check_interval, self._load_clip_prompt)

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

    def camera_info_cb(self, msg: CameraInfo):
        """ 
        Process camera intrinsic parameters 
        """
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(f"Camera intrinsics received: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

            self.pc_processor = PointCloudProcessor(
                self.fx, self.fy, self.cx, self.cy,
                device = self.device,
                depth_scale = self.depth_scale,
                pc_downsample = self.pc_downsample,
                pc_max_range = self.pc_max_range,
                )

    def depth_callback(self, msg: Image):
        """
        Store latest depth message.
        """
        with self.sync_lock:
            self.latest_depth_msg = msg

    def rgb_callback(self, msg: Image):
        """
        Process RGB image with synchronized depth.
        """
        with self.sync_lock:
            if self.latest_depth_msg is None:
                self.get_logger().debug("Waiting for depth message.")
                return
            rgb_msg = msg
            depth_msg = self.latest_depth_msg
        
        self.process_frame(rgb_msg, depth_msg)

    # --------------- Main Methods ------------- #

    def process_frame(self, rgb_msg: Image, depth_msg: Image):
        """
        Process a single RGB-D frame.
        """
        self.frame_count += 1

        cv_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        height, width = cv_depth.shape

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
        if not hasattr(res, 'masks') or len(res.boxes) == 0:
            return
        
        self._process_detections(res, cv_bgr, cv_depth, depth_msg, rgb_msg, height, width)

        if self.enable_vis:
            # result.plot() draws boxes, masks, and IDs on the image
            annotated_frame = res.plot()
            vis_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            vis_msg.header = rgb_msg.header
            self.vis_pub.publish(vis_msg)

    def _process_detections(self, res, cv_bgr, cv_depth, depth_msg, rgb_msg, height, width):
        """ 
        Unified pipeline: 
        1. Calculate Geometry (Centroids/PCs) for all objects.
        2. Publish PointCloud immediately.
        3. Batch Process CLIP for all objects.
        4. Publish Custom Messages with coupled Geometry + Embeddings.
        """
        xyxy = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.arange(len(clss))
        masks_t = res.masks.data if hasattr(res, 'masks') and res.masks is not None else None

        is_uint16 = (depth_msg.encoding == '16UC1')
        scale_factor = (1.0 / self.depth_scale) if is_uint16 else 1.0
        depth_t, valid_mask_t = self.pc_processor.prepare_depth_tensor(cv_depth, depth_msg.encoding, scale_factor)

        all_points_list = []
        frame_detections = []
        batch_queue = []

        do_clip_frame = (self.frame_count % max(1, self.frame_skip) == 0)

        for i in range(len(xyxy)):
            result = self.process_single_detection(
                i, xyxy[i], clss[i], ids[i], masks_t,
                cv_bgr, depth_t, valid_mask_t, scale_factor,
                height, width, rgb_msg, do_clip_frame
            )

            if result is None:
                continue

            instance_cloud, det_entry = result
            all_points_list.append(instance_cloud)
            frame_detections.append(det_entry)
            if det_entry.get("crop") is not None:
                batch_queue.append(det_entry)
            
        if all_points_list:
            final_points_t = torch.cat(all_points_list, dim=0)
            final_points = final_points_t.cpu().numpy().astype(np.float32)
            PointCloudProcessor.publish_pointcloud(final_points, depth_msg.header, self.pc_pub)
            self.get_logger().info(
                f"Published pointcloud with {final_points.shape[0]} points from {len(all_points_list)} instances."
            )

        if batch_queue:
            try:
                # Extract just the images for the batch
                images = [item["crop"] for item in batch_queue]
                
                # Run inference on all at once
                embeddings = self.clip.encode_images_batch(images)
                
                # Re-associate embeddings with metadata
                for i, emb in enumerate(embeddings):
                    batch_queue[i]["embedding"] = emb
                    # Clear the image to free memory
                    batch_queue[i]["crop"] = None
            except Exception as e:
                self.get_logger().error(f"Batch CLIP inference failed: {e}")

        self.publish_custom_detections(frame_detections, depth_msg.header, rgb_msg.header.stamp)
        self.publish_centroid_markers(frame_detections, depth_msg.header)

    def process_single_detection(self, idx, bbox, class_id, instance_id, masks_t,
                                 cv_bgr, depth_t, valid_mask_t, scale_factor,
                                 height, width, rgb_msg, do_clip_frame):
        """ 
        Process geometry and prepare data container. 
        Returns (pointcloud_tensor, detection_dictionary)
        """
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            return None
        
        if masks_t is None or idx >= masks_t.shape[0]:
            return None

        rgb_color = self.get_color_for_class(str(class_id), self.class_colors)
        instance_cloud_t, centroid = self.pc_processor.process_detection(
            masks_t[idx], depth_t, valid_mask_t,
            int(class_id), int(instance_id),
            rgb_color, scale_factor
        )

        if instance_cloud_t is None:
            return None
        
        # --- Create Shared Object (The "Box") ---
        class_name = self.class_id_to_name(int(class_id))

        detection_entry = {
            "class_id": int(class_id),
            "instance_id": int(instance_id),
            "object_name": class_name,
            "centroid": centroid, # (x, y, z) tuple
            "embedding": None,    # Placeholder
            "crop": None          # Placeholder
        }

        # --- Prepare CLIP Crop (If enabled) ---
        if do_clip_frame:
            sx1, sy1, sx2, sy2 = CLIPProcessor.compute_square_crop(
                x1, y1, x2, y2, width, height, self.square_crop_scale
            )
            if sx2 > sx1 and sy2 > sy1:
                # Build binary mask for this instance
                m = masks_t[idx].detach().cpu().numpy()
                if m.shape[0] != height or m.shape[1] != width:
                    m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
                binary_mask = (m > 0.5).astype(np.uint8) * 255

                img_crop = cv_bgr[sy1:sy2, sx1:sx2]
                mask_crop = binary_mask[sy1:sy2, sx1:sx2]

                if img_crop.size > 0 and mask_crop.size > 0:
                    # Gray masking
                    neutral_bg = np.full_like(img_crop, 122)  # neutral gray
                    bg_mask = cv2.bitwise_not(mask_crop)
                    fg = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
                    bg = cv2.bitwise_and(neutral_bg, neutral_bg, mask=bg_mask)
                    final_crop = cv2.add(fg, bg)
                    detection_entry["crop"] = final_crop

        return instance_cloud_t, detection_entry
    
    # ----------- Secondary Methods ------------ #

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

    def class_id_to_name(self, class_id: int) -> str:
        """
        Convert class ID to class name.
        """
        if 0 <= class_id < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id]
        return f"class_{class_id}"

    # ---------------- Publishers -------------- #

    def publish_custom_detections(self, detections, header, timestamp):
        """ Iterate through the unified list and publish messages. """
        for det in detections:
            msg = DetectedObject()
            
            # Basic Info
            msg.object_name = det["object_name"]
            msg.object_id = det["instance_id"]
            
            # Centroid
            cx, cy, cz = det["centroid"]
            msg.centroid = Vector3(x=cx, y=cy, z=cz)
            
            # Timestamp (Extract directly from the header)
            msg.timestamp = header.stamp
            
            current_emb = det["embedding"]
            msg.embedding = current_emb.tolist() if current_emb is not None else []

            # Only compute similarity if embedding is available
            prob_goal = 0.0
            prob_distractor = 0.0
            if current_emb is not None and self.goal_text_embedding is not None:
                prob_goal = self.clip.compute_sigmoid_probs(current_emb, self.goal_text_embedding)
                msg.similarity = float(prob_goal) if prob_goal is not None else 0.0
            else:
                msg.similarity = 0.0
                
            if current_emb is not None and self.distractor_embedding is not None:
                prob_distractor = self.clip.compute_sigmoid_probs(current_emb, self.distractor_embedding)

            self.get_logger().info(
                f"ID {det['instance_id']} ({det['object_name']}): Goal={prob_goal:.1%} | Distractor={prob_distractor:.1%}"
            )

            # Add text embedding for mapper convenience
            msg.text_embedding = self.goal_text_embedding.tolist() if self.goal_text_embedding is not None else []
            self.detections_pub.publish(msg)

    def publish_centroid_markers(self, detections, header):
        """Publish centroids as marker array."""
        try:
            marker_array = MarkerArray()
            
            for i, det in enumerate(detections):
                marker = Marker()
                marker.header = header
                marker.ns = "centroids"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # Set position from centroid
                cx, cy, cz = det["centroid"]
                marker.pose.position.x = cx
                marker.pose.position.y = cy
                marker.pose.position.z = cz
                marker.pose.orientation.w = 1.0
                
                # Set scale (radius of sphere)
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                
                # Set color from class color
                class_id_str = str(det["class_id"])
                r, g, b = self.get_color_for_class(class_id_str, self.class_colors)
                marker.color.r = r / 255.0
                marker.color.g = g / 255.0
                marker.color.b = b / 255.0
                marker.color.a = 1.0
                
                marker_array.markers.append(marker)
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing centroid markers: {e}")

# -------------------- MAIN -------------------- # 

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()