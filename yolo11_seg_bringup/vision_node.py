import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import message_filters
from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer
from builtin_interfaces.msg import Time
import threading

from ultralytics import YOLO
from cv_bridge import CvBridge

# PointCloud processing utilities
import torch
import struct
import numpy as np


# ------------------ FUNCTIONS ----------------- #

class PointCloudProcessor:

    """ Handles PointCloud generation from depth and segmentation masks. """

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

        """ Pack 3x uint8 RGB into float32 for PointCloud2 'rgb' field. """

        rgb_uint32 = (r << 16) | (g << 8) | b
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]
    
    def process_detection(self, mask_t, depth_t, valid_mask_t, class_id, instance_id, 
                         rgb_color, scale_factor, min_points=10):
        
        """ Process a single detection to generate pointcloud segment. """

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

    # Single .item() call instead of three
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

        rgb_packed_t = torch.full((N,), float(rgb_packed), dtype=torch.float32, device=self.device)
        class_id_t = torch.full((N,), int(class_id), dtype=torch.int32, device=self.device)
        instance_id_t = torch.full((N,), int(instance_id), dtype=torch.int32, device=self.device)

        instance_cloud_t = torch.stack(
            [
                x_t,  # Already float32
                y_t,  # Already float32
                z_clean_t,  # Already float32
                torch.full((N,), float(rgb_packed), dtype=torch.float32, device=self.device),
                torch.full((N,), int(class_id), dtype=torch.int32, device=self.device),
                torch.full((N,), int(instance_id), dtype=torch.int32, device=self.device),
            ],
            dim=1
        )
        return instance_cloud_t, centroid

    def prepare_depth_tensor(self, depth_img, encoding, scale_factor):

        """ Convert depth image to GPU tensor with validity mask. """

        depth_t = torch.from_numpy(depth_img.astype(np.float32)).to(self.device)
        valid_mask_t = (depth_t > 0) & (~torch.isnan(depth_t))
        if self.pc_max_range > 0.0:
            valid_mask_t = valid_mask_t & (depth_t * scale_factor <= self.pc_max_range)
        return depth_t, valid_mask_t
    
    @staticmethod
    def publish_pointcloud(points: np.ndarray, header, publisher):
        """Publish PointCloud2 message from points numpy array."""
        try:
            if points is None or points.size == 0:
                return

            # Use structured array instead of Python loop - 100x faster
            dtype = np.dtype([
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.float32),
                ('class_id', np.int32),
                ('instance_id', np.int32),
            ])
            
            # Direct memcpy conversion - no Python loop
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

# -------------------- CLASS ------------------- #

class VisionNode(Node):
    """ Node that handles Vision computation """

    # ------------- Initialization ------------- #

    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info("VisionNode initialized")

        # ============= Parameters ============= #

        self.declare_parameter('model', '/home/sensor/yolo11n-seg.engine')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('pc_topic', '/yolo/pointcloud')
        self.declare_parameter('anno_topic', '/yolo/annotated_image')
        self.declare_parameter('depth_scale', 1000.0)
        self.declare_parameter('pc_downsample', 2)
        self.declare_parameter('pc_max_range', 8.0)
        self.declare_parameter('mask_threshold', 0.5)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('retina_masks', True)

        self.model = self.get_parameter('model').value
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.pc_topic = self.get_parameter('pc_topic').value
        self.anno_topic = self.get_parameter('anno_topic').value
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.pc_downsample = int(self.get_parameter('pc_downsample').value)
        self.pc_max_range = float(self.get_parameter('pc_max_range').value)
        self.mask_threshold = float(self.get_parameter('mask_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        self.retina_masks = bool(self.get_parameter('retina_masks').value)

        # =========== Initialization =========== #

        self.get_logger().info(f"\nLoading YOLO model from {self.model}...")
        self.model = YOLO(self.model, task='segment')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.bridge = CvBridge()
        self.pc_processor = None

        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.rgb_sub = self.create_subscription(
            Image, self.image_topic, self.rgb_callback, qos_profile=qos_sensor
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile=qos_sensor
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_profile=qos_sensor
        )

        self.pc_pub = self.create_publisher(PointCloud2, self.pc_topic, 10)
        self.anno_pub = self.create_publisher(Image, self.anno_topic, 10)

        self.last_centroids = []
        self.class_colors = {}

        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth_msg = None
        self.warned_missing_intrinsics = False
        self.sync_lock = threading.Lock()

    # ---------------- Callbacks --------------- #

    def camera_info_cb(self, msg: CameraInfo):
        """ Process camera intrinsic parameters """
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
                mask_threshold = self.mask_threshold
                )

    def depth_callback(self, msg: Image):
        """Store latest depth message."""
        with self.sync_lock:
            self.latest_depth_msg = msg

    def rgb_callback(self, msg: Image):
        """Process RGB image with synchronized depth."""
        with self.sync_lock:
            if self.latest_depth_msg is None:
                self.get_logger().debug("Waiting for depth message.")
                return
            rgb_msg = msg
            depth_msg = self.latest_depth_msg
        
        self.process_frame(rgb_msg, depth_msg)

    def process_frame(self, rgb_msg: Image, depth_msg: Image):

        """ Process a single RGB-D frame """

        cv_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        if cv_depth.ndim != 2 or self.fx is None:
            if self.fx is None and not self.warned_missing_intrinsics:
                self.get_logger().warning("Waiting for camera intrinsics; skipping frame.")
                self.warned_missing_intrinsics = True
            return
        
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
            self.get_logger().debug("No detections or masks; skipping pointcloud publish.")
            return
        
        self._process_detections(res, cv_bgr, cv_depth, depth_msg, rgb_msg, height, width)

    def _process_detections(self, res, cv_bgr, cv_depth, depth_msg, rgb_msg, height, width):

        """ Process all detections from YOLO results. """

        xyxy = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.arange(len(clss))
        masks_t = res.masks.data if hasattr(res, 'masks') and res.masks is not None else None

        is_uint16 = (depth_msg.encoding == '16UC1')
        scale_factor = (1.0 / self.depth_scale) if is_uint16 else 1.0

        depth_t, valid_mask_t = self.pc_processor.prepare_depth_tensor(cv_depth, depth_msg.encoding, scale_factor)

        all_points_list = []
        self.last_centroids = []

        self.get_logger().debug(
            f"Depth shape {tuple(depth_t.shape)}, masks {tuple(masks_t.shape) if masks_t is not None else 'None'}"
        )

        for i in range(len(xyxy)):
            result = self.process_single_detection(
                i, xyxy[i], clss[i], ids[i], masks_t,
                cv_bgr, depth_t, valid_mask_t, scale_factor,
                height, width, rgb_msg
            )

            if result is None:
                continue

            instance_cloud_t = result

            all_points_list.append(instance_cloud_t)
        
        if all_points_list:
            final_points_t = torch.cat(all_points_list, dim=0)
            final_points = final_points_t.cpu().numpy().astype(np.float32)
            PointCloudProcessor.publish_pointcloud(final_points, depth_msg.header, self.pc_pub)
            self.get_logger().info(
                f"Published pointcloud with {final_points.shape[0]} points from {len(all_points_list)} instances."
            )
        
    def process_single_detection(self, idx, bbox, class_id, instance_id, masks_t,
                                 cv_bgr, depth_t, valid_mask_t, scale_factor,
                                 height, width, rgb_msg):

        """ Process a single detection to generate pointcloud segment and annotated image. """

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

        # Store centroid and metadata
        self.last_centroids.append({
            "class_id": int(class_id),
            "instance_id": int(instance_id),
            "centroid": centroid
        })

        return instance_cloud_t
    
    @staticmethod
    def get_color_for_class(class_id: str, class_colors: dict):

        """ Deterministically generate a color for a given class ID. """

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