import struct

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import tf2_ros
from rclpy.time import Time

from yolo11_seg_interfaces.msg import DetectedObjectV2


class DetectionPointsCloudNode(Node):
    """Republish all active DetectedObjectV2.object_points as a single stable PointCloud2 in the map frame."""

    def __init__(self) -> None:
        super().__init__('detection_points_cloud_node')

        self.declare_parameter('detection_topic', '/vision/detections')
        self.declare_parameter('pointcloud_topic', '/vision/detected_object_points')
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('object_ttl_sec', 1.0)
        self.declare_parameter('publish_rate_hz', 8.0)
        self.declare_parameter('tf_timeout_sec', 0.1)

        self.detection_topic = str(self.get_parameter('detection_topic').value)
        self.pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.object_ttl_sec = float(self.get_parameter('object_ttl_sec').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        if self.object_ttl_sec <= 0.0:
            self.object_ttl_sec = 1.0
        if self.publish_rate_hz <= 0.0:
            self.publish_rate_hz = 8.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pc_pub = self.create_publisher(PointCloud2, self.pointcloud_topic, 10)
        self.det_sub = self.create_subscription(
            DetectedObjectV2,
            self.detection_topic,
            self.detection_cb,
            10,
        )
        self.pub_timer = self.create_timer(1.0 / self.publish_rate_hz, self.publish_cached_cloud)

        # object_id -> {'points_xyz': [[x,y,z], ...] already in map frame, 'last_seen_ns': int, 'name': str}
        self.object_cache = {}

        self.get_logger().info(
            f'Subscribed to {self.detection_topic}, publishing cloud on {self.pointcloud_topic}, '
            f'camera_frame={self.camera_frame} -> map_frame={self.map_frame}, '
            f'ttl={self.object_ttl_sec}s, rate={self.publish_rate_hz}Hz'
        )

    @staticmethod
    def _pack_rgb_float(r: int, g: int, b: int) -> float:
        """Pack 8-bit RGB channels into the float32 format expected by PointCloud2."""
        rgb_uint32 = (r << 16) | (g << 8) | b
        return struct.unpack('f', struct.pack('I', rgb_uint32))[0]

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _transform_points(points_xyz: list, transform) -> list:
        """Apply a geometry_msgs/TransformStamped to a list of [x,y,z] points."""
        t = transform.transform.translation
        r = transform.transform.rotation
        # Build rotation matrix from quaternion (x, y, z, w)
        qx, qy, qz, qw = r.x, r.y, r.z, r.w
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)
        pts = np.array(points_xyz, dtype=np.float64)   # (N, 3)
        transformed = (R @ pts.T).T + translation       # (N, 3)
        return transformed.tolist()

    def detection_cb(self, msg: DetectedObjectV2) -> None:
        if len(msg.object_points) == 0:
            return

        points_cam = [[float(p.x), float(p.y), float(p.z)] for p in msg.object_points]
        stamp = msg.timestamp
        seen_ns = self._stamp_to_ns(stamp)

        # Look up camera -> map transform at the detection's timestamp.
        try:
            tf_stamp = Time(seconds=stamp.sec, nanoseconds=stamp.nanosec)
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                tf_stamp,
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF lookup {self.camera_frame}->{self.map_frame} failed: {e}. "
                "Trying latest available transform.",
                throttle_duration_sec=2.0,
            )
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    Time(),  # latest
                    timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
                )
            except Exception as e2:
                self.get_logger().error(
                    f"TF latest lookup also failed: {e2}. Dropping detection.",
                    throttle_duration_sec=2.0,
                )
                return

        points_map = self._transform_points(points_cam, transform)

        self.object_cache[int(msg.object_id)] = {
            'points_xyz': points_map,
            'last_seen_ns': seen_ns,
            'name': str(msg.object_name),
        }

        self.get_logger().info(
            f"Cached {len(points_map)} points (map frame) for object_id={msg.object_id} ({msg.object_name})",
            throttle_duration_sec=2.0,
        )

    def publish_cached_cloud(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        ttl_ns = int(self.object_ttl_sec * 1e9)

        stale_ids = [
            obj_id
            for obj_id, item in self.object_cache.items()
            if (now_ns - int(item['last_seen_ns'])) > ttl_ns
        ]
        for obj_id in stale_ids:
            del self.object_cache[obj_id]

        if not self.object_cache:
            return

        points = []
        for obj_id, item in self.object_cache.items():
            r = (obj_id * 73) % 255
            g = (obj_id * 151) % 255
            b = (obj_id * 199) % 255
            rgb = self._pack_rgb_float(r, g, b)

            for x, y, z in item['points_xyz']:
                points.append([x, y, z, rgb])

        if not points:
            return

        header = Header()
        header.frame_id = self.map_frame
        header.stamp = self.get_clock().now().to_msg()

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = point_cloud2.create_cloud(header, fields, points)
        self.pc_pub.publish(cloud_msg)

        self.get_logger().info(
            f'Published merged cloud: objects={len(self.object_cache)}, points={len(points)}',
            throttle_duration_sec=2.0,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionPointsCloudNode()
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
