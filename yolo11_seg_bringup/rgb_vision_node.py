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
import clip
from PIL import Image as PILImage

# Import your custom interface
from yolo11_seg_interfaces.msg import DetectedObject

class CLIPProcessor:
    """Handles CLIP model loading and embedding generation."""
    def __init__(self, model_name='ViT-B/32', device='cpu'):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

    def encode_images_batch(self, crops_bgr):
        """Encode a batch of BGR image crops to CLIP embeddings."""
        if not crops_bgr:
            return []
        
        # Preprocess and stack
        pil_images = [PILImage.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        image_input = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()

    @staticmethod
    def compute_square_crop(x1, y1, x2, y2, width, height, scale=1.3):
        """Compute square crop coordinates for CLIP."""
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
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('clip_every_n_frames', 3) # Optimization: don't run CLIP every frame

        self.model_path = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.anno_topic = self.get_parameter('annotated_topic').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.conf_thres = self.get_parameter('conf').value
        self.clip_skip = self.get_parameter('clip_every_n_frames').value

        # --- Setup ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # Load Models
        self.get_logger().info(f"Loading YOLO: {self.model_path}")
        self.yolo = YOLO(self.model_path, task='segment')
        
        self.get_logger().info("Loading CLIP...")
        self.clip = CLIPProcessor(device=self.device)

        # ROS Comm
        self.bridge = CvBridge()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.sub_img = self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.pub_det = self.create_publisher(DetectedObject, self.detections_topic, 10)
        self.pub_anno = self.create_publisher(Image, self.anno_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.markers_topic, 10)

        self.frame_count = 0
        self.get_logger().info("RGB Vision Node Ready.")

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

            # 3. Process Detections
            for i, box in enumerate(boxes):
                # Skip class 0
                if classes[i] == 0:
                    continue
                    
                x1, y1, x2, y2 = box
                
                # Calculate 2D Centroid from Mask (if available) or BBox
                cx, cy = 0.0, 0.0
                if masks is not None:
                    # Resize mask to original image size if needed, but YOLO usually returns scaled masks
                    # Simple moment calculation on the binary mask
                    m = masks[i].cpu().numpy().astype(np.uint8)
                    M = cv2.moments(m)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) # Scale if mask is smaller than image
                        cy = int(M["m01"] / M["m00"])
                        # Note: If retimask=True, mask matches image size usually
                        # If mask is downsampled, you must scale cx, cy up.
                        # Assuming retina_masks=True for 1:1 mapping:
                        cx = float(cx * (w / m.shape[1]))
                        cy = float(cy * (h / m.shape[0]))
                    else:
                        cx, cy = float((x1+x2)/2), float((y1+y2)/2)
                else:
                    cx, cy = float((x1+x2)/2), float((y1+y2)/2)

                # Prepare CLIP crop
                sx1, sy1, sx2, sy2 = CLIPProcessor.compute_square_crop(x1, y1, x2, y2, w, h)
                crop = img[sy1:sy2, sx1:sx2]
                
                if crop.size == 0: continue

                detections_to_process.append({
                    "id": ids[i],
                    "class": self.yolo.names[classes[i]],
                    "centroid": (cx, cy),
                    "bbox": box
                })
                crops_for_clip.append(crop)

            # 4. Batch CLIP Inference (Run less frequently for speed)
            embeddings = []
            if self.frame_count % self.clip_skip == 0 and crops_for_clip:
                embeddings = self.clip.encode_images_batch(crops_for_clip)

            # 5. Publish DetectedObjects and Markers
            marker_array = MarkerArray()
            for i, det in enumerate(detections_to_process):
                msg_det = DetectedObject()
                msg_det.timestamp = msg.header.stamp
                msg_det.object_id = int(det["id"])
                msg_det.object_name = det["class"]
                
                # 2D Centroid in a 3D Vector (Z=0)
                msg_det.centroid = Vector3(x=det["centroid"][0], y=det["centroid"][1], z=0.0)
                
                # Attach embedding if computed
                if i < len(embeddings):
                    msg_det.embedding = embeddings[i].tolist()
                
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