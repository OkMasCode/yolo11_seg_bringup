import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


class StillImagePublisherNode(Node):
    """Publishes a still image repeatedly as sensor_msgs/Image."""

    def __init__(self):
        super().__init__("still_image_publisher_node")

        self.declare_parameter("image_path", "/workspaces/wine_bottles.jpg")
        self.declare_parameter("image_topic", "/camera/rgb")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("frame_id", "camera_rgb_optical_frame")
        self.declare_parameter("output_height", 360)
        self.declare_parameter("output_width", 480)

        self.image_path = str(self.get_parameter("image_path").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.fps = float(self.get_parameter("fps").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.output_height = int(self.get_parameter("output_height").value)
        self.output_width = int(self.get_parameter("output_width").value)

        if self.fps <= 0.0:
            self.get_logger().warn("Invalid fps <= 0. Falling back to 30.0")
            self.fps = 30.0

        if self.output_height <= 0 or self.output_width <= 0:
            self.get_logger().warn(
                "Invalid output size. Falling back to 360x480 (HxW)"
            )
            self.output_height = 360
            self.output_width = 480

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image path does not exist: {self.image_path}")

        self.cv_bgr = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if self.cv_bgr is None or self.cv_bgr.size == 0:
            raise RuntimeError(f"Failed to decode image: {self.image_path}")

        # Resize once so every published frame is fixed at the requested dimensions.
        self.cv_bgr = cv2.resize(
            self.cv_bgr,
            (self.output_width, self.output_height),
            interpolation=cv2.INTER_AREA,
        )

        self.bridge = CvBridge()
        self.publish_count = 0

        qos_sensor = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.image_pub = self.create_publisher(Image, self.image_topic, qos_sensor)
        self.timer = self.create_timer(1.0 / self.fps, self._publish_image)

        h, w = self.cv_bgr.shape[:2]
        self.get_logger().info(
            "Still image publisher ready: "
            f"path='{self.image_path}', topic='{self.image_topic}', fps={self.fps:.2f}, "
            f"size={w}x{h} (WxH), encoding='bgr8'"
        )

    def _publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(self.cv_bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.image_pub.publish(msg)

        self.publish_count += 1
        if self.publish_count % 60 == 0:
            self.get_logger().info(
                f"Published {self.publish_count} frames to {self.image_topic}",
                throttle_duration_sec=2.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = StillImagePublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Publisher node startup failed: {exc}")
        else:
            print(f"Publisher node startup failed: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()