"""Publish a static RGB image and dummy aligned depth at a fixed rate."""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge


class BusImagePublisherNode(Node):
    """ROS 2 node that republishes a local image as camera RGB + aligned depth."""

    def __init__(self):
        super().__init__('bus_image_publisher_node')

        self.declare_parameter('image_path', '/home/workspace/bus.jpg')
        self.declare_parameter('rgb_topic', '/jackal/sensors/camera_0/color/image')
        self.declare_parameter('depth_topic', '/jackal/sensors/camera_0/aligned_depth_to_color/image')
        self.declare_parameter('frame_id', 'camera_0_color_optical_frame')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('dummy_depth_m', 2.0)

        self.image_path = str(self.get_parameter('image_path').value)
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.dummy_depth_m = float(self.get_parameter('dummy_depth_m').value)

        bgr = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().error(f"Failed to load image: '{self.image_path}'")
            raise RuntimeError('Image load failed')

        self.height, self.width = bgr.shape[:2]
        dummy_depth_mm = int(max(0.0, self.dummy_depth_m) * 1000.0)
        self.depth_u16 = np.full((self.height, self.width), dummy_depth_mm, dtype=np.uint16)

        self.bridge = CvBridge()
        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.rgb_pub = self.create_publisher(Image, self.rgb_topic, qos_sensor)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, qos_sensor)

        self.rgb_msg_template = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        self.depth_msg_template = self.bridge.cv2_to_imgmsg(self.depth_u16, encoding='16UC1')

        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._publish)

        self.get_logger().info(
            f"Publishing static image '{self.image_path}' at {self.rate_hz:.1f} Hz\n"
            f"RGB topic: {self.rgb_topic}\n"
            f"Depth topic: {self.depth_topic}\n"
            f"Image size: {self.width}x{self.height}, dummy depth: {dummy_depth_mm} mm"
        )

    def _publish(self):
        now = self.get_clock().now().to_msg()

        rgb_msg = Image()
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = self.frame_id
        rgb_msg.height = self.rgb_msg_template.height
        rgb_msg.width = self.rgb_msg_template.width
        rgb_msg.encoding = self.rgb_msg_template.encoding
        rgb_msg.is_bigendian = self.rgb_msg_template.is_bigendian
        rgb_msg.step = self.rgb_msg_template.step
        rgb_msg.data = self.rgb_msg_template.data

        depth_msg = Image()
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = self.frame_id
        depth_msg.height = self.depth_msg_template.height
        depth_msg.width = self.depth_msg_template.width
        depth_msg.encoding = self.depth_msg_template.encoding
        depth_msg.is_bigendian = self.depth_msg_template.is_bigendian
        depth_msg.step = self.depth_msg_template.step
        depth_msg.data = self.depth_msg_template.data

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BusImagePublisherNode()
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
