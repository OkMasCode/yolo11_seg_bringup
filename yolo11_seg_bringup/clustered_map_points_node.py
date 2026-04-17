import json
import colorsys
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class ClusteredMapPointsNode(Node):
    def __init__(self) -> None:
        super().__init__('clustered_map_points_node')

        self.declare_parameter(
            'clustered_map_file',
            '/workspaces/ros2_ws/src/yolo11_seg_bringup/config/clustered_map_v6.json',
        )
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('marker_topic', '/vision/clustered_map_objects_markers')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('point_scale', 0.15)
        self.declare_parameter('label_height_offset', 0.2)

        self.clustered_map_file = str(self.get_parameter('clustered_map_file').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.point_scale = float(self.get_parameter('point_scale').value)
        self.label_height_offset = float(self.get_parameter('label_height_offset').value)

        if self.publish_rate_hz <= 0.0:
            self.publish_rate_hz = 1.0

        self.markers_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.publish_markers)

        self.get_logger().info(
            f'ClusteredMapPointsNode started. clustered_map_file={self.clustered_map_file}, '
            f'frame={self.map_frame}, topic={self.marker_topic}'
        )

    @staticmethod
    def cluster_to_color(cluster_id: int) -> ColorRGBA:
        # Keep color stable and distinct per cluster id.
        hue = ((cluster_id * 37) % 360) / 360.0
        saturation = 0.85
        value = 0.95
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        return ColorRGBA(r=float(red), g=float(green), b=float(blue), a=1.0)

    def load_clustered_entries(self) -> list:
        path = Path(self.clustered_map_file)
        if not path.exists():
            self.get_logger().warn(f'Clustered map file not found: {self.clustered_map_file}')
            return []

        try:
            with path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                self.get_logger().warn(
                    f'Expected list in clustered map file, got: {type(data).__name__}'
                )
                return []
            return data
        except Exception as exc:
            self.get_logger().error(f'Failed to read clustered map file {self.clustered_map_file}: {exc}')
            return []

    def publish_markers(self) -> None:
        map_entries = self.load_clustered_entries()

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        points_marker = Marker()
        points_marker.header.frame_id = self.map_frame
        points_marker.header.stamp = now
        points_marker.ns = 'clustered_map_objects_points'
        points_marker.id = 0
        points_marker.type = Marker.SPHERE_LIST
        points_marker.action = Marker.ADD
        points_marker.scale.x = self.point_scale
        points_marker.scale.y = self.point_scale
        points_marker.scale.z = self.point_scale

        text_markers = []
        text_id = 1

        for entry in map_entries:
            if not isinstance(entry, dict):
                continue

            coords = entry.get('coords', {})
            if not isinstance(coords, dict):
                continue

            if not all(k in coords for k in ('x', 'y')):
                continue

            try:
                point_x = float(coords['x'])
                point_y = float(coords['y'])
                point_z = float(coords.get('z', 0.0))
                cluster_id = int(entry.get('cluster', -1))
            except (TypeError, ValueError):
                continue

            point = Point(x=point_x, y=point_y, z=point_z)
            points_marker.points.append(point)

            class_name = str(entry.get('class', 'unknown'))
            object_id = str(entry.get('id', ''))
            class_color = self.cluster_to_color(cluster_id)
            points_marker.colors.append(class_color)

            label_parts = [class_name, f'cluster:{cluster_id}']
            if object_id:
                label_parts.append(object_id)
            label = ' | '.join(label_parts)

            text_marker = Marker()
            text_marker.header.frame_id = self.map_frame
            text_marker.header.stamp = now
            text_marker.ns = 'clustered_map_objects_labels'
            text_marker.id = text_id
            text_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = point.x
            text_marker.pose.position.y = point.y
            text_marker.pose.position.z = point.z + self.label_height_offset
            text_marker.scale.z = 0.2
            text_marker.color = class_color
            text_marker.text = label
            text_markers.append(text_marker)

        marker_array.markers.append(points_marker)
        marker_array.markers.extend(text_markers)
        self.markers_pub.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusteredMapPointsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
