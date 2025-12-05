#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import numpy as np

class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud_node')

        self.depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        self.info_topic = '/camera/camera/color/camera_info' 

        self.depth_scale = 0.001  # RealSense uint16 mm -> meters
        self.decimation = 2       # 1 = Full Res, 2 = Half Res (Faster)

        # --- SETUP ---
        self.bridge = CvBridge()
        self.camera_info = None
        self.fx = self.fy = self.cx = self.cy = None

        self.pub_pc = self.create_publisher(PointCloud2, '/output/pointcloud', 10)

        # Subscriptions
        self.create_subscription(CameraInfo, self.info_topic, self.info_cb, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, 10)

        self.get_logger().info(f"Listening on {self.depth_topic}...")

    def info_cb(self, msg):
        """Get camera intrinsics once."""
        if self.camera_info is None:
            self.camera_info = msg
            K = np.array(msg.k).reshape(3, 3)
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
            self.get_logger().info(f"Intrinsics Loaded: {self.fx}, {self.fy}")

    def depth_cb(self, msg):
        """Process depth immediately upon arrival."""
        if self.camera_info is None:
            self.get_logger().warn_throttle(2.0, "Waiting for Camera Info...")
            return

        try:
            # passthrough preserves uint16 (mm)
            depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # --- VECTORIZED PROJECTION ---
        
        # 1. Downsample (Optional optimization)
        step = self.decimation
        if step > 1:
            depth_img = depth_img[::step, ::step] # Downsample by skipping rows/cols
            # Adjust intrinsics
            fx = self.fx / step
            fy = self.fy / step
            cx = self.cx / step
            cy = self.cy / step
        else:
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        # 2. Convert to meters
        # Filter 0 (invalid) and huge values (noise)
        depth_m = depth_img.astype(np.float32) * self.depth_scale
        valid = (depth_m > 0) & (depth_m < 10.0) 
        
        # 3. Create Grid
        H, W = depth_img.shape
        u_indices = np.arange(W)
        v_indices = np.arange(H)
        u_grid, v_grid = np.meshgrid(u_indices, v_indices)

        # 4. Project pixels to 3D
        # Keep only valid points to save memory
        z = depth_m[valid]
        u = u_grid[valid]
        v = v_grid[valid]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # 5. Stack (N, 3) -> x, y, z
        points = np.column_stack((x, y, z))

        if points.shape[0] == 0:
            return

        # 6. Publish
        # We only need XYZ fields now
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pc2 = point_cloud2.create_cloud(msg.header, fields, points)
        self.pub_pc.publish(pc2)


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()