import json
import colorsys
import hashlib
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class MapPointsNode(Node):
    def __init__(self) -> None:
        super().__init__('map_points_node')

        self.declare_parameter(
            'map_file',
            '/workspaces/ros2_ws/src/yolo11_seg_bringup/config/map_v5.json',
        )
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('marker_topic', '/vision/map_objects_markers')
        self.declare_parameter('bbox_marker_topic', '/vision/map_objects_bbox_markers')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('point_scale', 0.15)
        self.declare_parameter('label_height_offset', 0.2)
        self.declare_parameter('bbox_alpha', 0.35)
        self.declare_parameter('bbox_min_edge_m', 0.02)
        self.declare_parameter('bbox_line_width', 0.03)

        self.map_file = str(self.get_parameter('map_file').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.bbox_marker_topic = str(self.get_parameter('bbox_marker_topic').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.point_scale = float(self.get_parameter('point_scale').value)
        self.label_height_offset = float(self.get_parameter('label_height_offset').value)
        self.bbox_alpha = float(self.get_parameter('bbox_alpha').value)
        self.bbox_min_edge_m = float(self.get_parameter('bbox_min_edge_m').value)
        self.bbox_line_width = float(self.get_parameter('bbox_line_width').value)

        if self.publish_rate_hz <= 0.0:
            self.publish_rate_hz = 1.0
        if self.bbox_alpha <= 0.0 or self.bbox_alpha > 1.0:
            self.bbox_alpha = 0.35
        if self.bbox_min_edge_m <= 0.0:
            self.bbox_min_edge_m = 0.02
        if self.bbox_line_width <= 0.0:
            self.bbox_line_width = 0.03

        self.markers_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.bbox_markers_pub = self.create_publisher(MarkerArray, self.bbox_marker_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.publish_markers)

        self.get_logger().info(
            f'MapPointsNode started. map_file={self.map_file}, frame={self.map_frame}, '
            f'topic={self.marker_topic}, bbox_topic={self.bbox_marker_topic}'
        )

    def compute_marker_z(self, pose_map: dict) -> float:
        try:
            return float(pose_map.get('z', 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def class_to_color(class_name: str) -> ColorRGBA:
        digest = hashlib.md5(class_name.encode('utf-8')).digest()
        hue = digest[0] / 255.0
        saturation = 0.8
        value = 0.95
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        return ColorRGBA(r=float(red), g=float(green), b=float(blue), a=1.0)

    def load_map_entries(self) -> dict:
        path = Path(self.map_file)
        if not path.exists():
            self.get_logger().warn(f'Map file not found: {self.map_file}')
            return {}

        try:
            with path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                self.get_logger().warn(f'Expected dict in map file, got: {type(data).__name__}')
                return {}
            self.get_logger().info(
                f'[map_points] loaded map entries: {len(data)}',
                throttle_duration_sec=5.0,
            )
            return data
        except Exception as exc:
            self.get_logger().error(f'Failed to read map file {self.map_file}: {exc}')
            return {}

    def publish_markers(self) -> None:
        map_entries = self.load_map_entries()

        marker_array = MarkerArray()
        bbox_marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        clear_bbox_marker = Marker()
        clear_bbox_marker.action = Marker.DELETEALL
        bbox_marker_array.markers.append(clear_bbox_marker)

        points_marker = Marker()
        points_marker.header.frame_id = self.map_frame
        points_marker.header.stamp = now
        points_marker.ns = 'map_objects_points'
        points_marker.id = 0
        points_marker.type = Marker.SPHERE_LIST
        points_marker.action = Marker.ADD
        points_marker.scale.x = self.point_scale
        points_marker.scale.y = self.point_scale
        points_marker.scale.z = self.point_scale
        points_marker.color.r = 0.1
        points_marker.color.g = 0.7
        points_marker.color.b = 1.0
        points_marker.color.a = 1.0

        text_markers = []
        text_id = 1

        valid_entries = []
        for map_id, entry in map_entries.items():
            if not isinstance(entry, dict):
                continue

            pose_map = entry.get('pose_map', {})
            if not isinstance(pose_map, dict):
                continue

            if not all(k in pose_map for k in ('x', 'y')):
                continue

            try:
                point_x = float(pose_map['x'])
                point_y = float(pose_map['y'])
                marker_z = self.compute_marker_z(pose_map)
            except (TypeError, ValueError):
                continue

            valid_entries.append((map_id, entry, point_x, point_y, marker_z))

        self.get_logger().info(
            f'[map_points] valid entries for publish: {len(valid_entries)} / {len(map_entries)}',
            throttle_duration_sec=5.0,
        )

        # Open3D box corner order expected from export_to_json:
        # 0:(min,min,min), 1:(max,min,min), 2:(min,max,min), 3:(min,min,max),
        # 4:(max,max,max), 5:(min,max,max), 6:(max,min,max), 7:(max,max,min)
        # Edges pair these corners into a 12-edge wireframe box.
        edge_pairs = (
            (0, 1), (1, 7), (7, 2), (2, 0),
            (3, 6), (6, 4), (4, 5), (5, 3),
            (0, 3), (1, 6), (2, 5), (7, 4),
        )

        for index, (map_id, entry, point_x, point_y, marker_z) in enumerate(valid_entries):
            point_z = marker_z
            point = Point(
                x=point_x,
                y=point_y,
                z=point_z,
            )

            points_marker.points.append(point)

            class_name = str(entry.get('name', 'unknown'))
            occurrences = int(entry.get('occurrences', 1))
            label = f'{class_name} (x{occurrences})'
            class_color = self.class_to_color(class_name)
            points_marker.colors.append(class_color)

            text_marker = Marker()
            text_marker.header.frame_id = self.map_frame
            text_marker.header.stamp = now
            text_marker.ns = 'map_objects_labels'
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

            box_size = entry.get('box_size', {})
            if isinstance(box_size, dict):
                sx = float(box_size.get('x', 0.0) or 0.0)
                sy = float(box_size.get('y', 0.0) or 0.0)
                sz = float(box_size.get('z', 0.0) or 0.0)
            else:
                sx = 0.0
                sy = 0.0
                sz = 0.0

            if sx > 0.0 and sy > 0.0 and sz > 0.0:
                corners = []
                raw_corners = entry.get('bbox_corners', [])
                if isinstance(raw_corners, list) and len(raw_corners) == 8:
                    for raw_corner in raw_corners:
                        if not isinstance(raw_corner, dict):
                            corners = []
                            break
                        try:
                            corners.append(
                                Point(
                                    x=float(raw_corner.get('x', 0.0)),
                                    y=float(raw_corner.get('y', 0.0)),
                                    z=float(raw_corner.get('z', 0.0)),
                                )
                            )
                        except (TypeError, ValueError):
                            corners = []
                            break

                # Backward-compatible fallback for older maps without explicit OBB corners.
                if len(corners) != 8:
                    hx = max(sx, self.bbox_min_edge_m) * 0.5
                    hy = max(sy, self.bbox_min_edge_m) * 0.5
                    hz = max(sz, self.bbox_min_edge_m) * 0.5
                    cx = point.x
                    cy = point.y
                    cz = point.z
                    corners = [
                        Point(x=cx - hx, y=cy - hy, z=cz - hz),  # 0
                        Point(x=cx + hx, y=cy - hy, z=cz - hz),  # 1
                        Point(x=cx - hx, y=cy + hy, z=cz - hz),  # 2
                        Point(x=cx - hx, y=cy - hy, z=cz + hz),  # 3
                        Point(x=cx + hx, y=cy + hy, z=cz + hz),  # 4
                        Point(x=cx - hx, y=cy + hy, z=cz + hz),  # 5
                        Point(x=cx + hx, y=cy - hy, z=cz + hz),  # 6
                        Point(x=cx + hx, y=cy + hy, z=cz - hz),  # 7
                    ]

                bbox_marker = Marker()
                bbox_marker.header.frame_id = self.map_frame
                bbox_marker.header.stamp = now
                bbox_marker.ns = 'map_objects_bbox'
                bbox_marker.id = index
                bbox_marker.type = Marker.LINE_LIST
                bbox_marker.action = Marker.ADD
                bbox_marker.pose.orientation.w = 1.0
                bbox_marker.scale.x = self.bbox_line_width
                bbox_marker.color = class_color
                bbox_marker.color.a = self.bbox_alpha
                bbox_marker.text = str(map_id)

                for edge_start, edge_end in edge_pairs:
                    bbox_marker.points.append(corners[edge_start])
                    bbox_marker.points.append(corners[edge_end])

                bbox_marker_array.markers.append(bbox_marker)

        marker_array.markers.append(points_marker)
        marker_array.markers.extend(text_markers)
        self.markers_pub.publish(marker_array)
        self.bbox_markers_pub.publish(bbox_marker_array)
        self.get_logger().info(
            f'[map_points] published markers: points={len(points_marker.points)}, labels={len(text_markers)}, '
            f'bboxes={max(len(bbox_marker_array.markers) - 1, 0)}',
            throttle_duration_sec=5.0,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapPointsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()