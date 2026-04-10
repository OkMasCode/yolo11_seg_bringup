import json
from pathlib import Path
from collections import Counter
from typing import List

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

# Adjust if your messages are in a different package
from yolo11_seg_interfaces.msg import (
    ClusterBoundingBox2D,
    ClusterDimensions2D,
    ClusteredMapObject,
    ClusteredMapObjectArray,
    SemanticObjectArray
)


class ClusteredMapPreprocPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("clustered_map_preproc_publisher_node")

        self.declare_parameter("output_clustered_map_file", "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/clustered_map_v6.json")
        self.declare_parameter("output_topic", "/vision/clustered_map_v6")
        self.declare_parameter("input_topic", "/vision/semantic_map_v5")
        self.declare_parameter("pointcloud_topic", "/vision/semantic_map_v5/points")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 0.5)
        
        # --- TUNING PARAMETERS ---
        self.declare_parameter("dist_thresh_multiplier", 0.5)
        self.declare_parameter("wall_search_radius_px", 10) 
        
        # NEW: Dilation kernel size. Since pointclouds have gaps between points, 
        # we dilate the footprint by this many pixels to create a solid eraser stamp.
        # 5 pixels * 0.05m = ~25cm dilation.
        self.declare_parameter("pc_dilation_px", 8) 
        self.declare_parameter("enable_object_removal", True)

        self.output_clustered_map_file = str(self.get_parameter("output_clustered_map_file").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.pc_topic = str(self.get_parameter("pointcloud_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        # State Variables
        self.latest_map_msg = None
        self.latest_semantic_msg = None
        self.latest_pc_msg = None
        self.room_markers = None
        self.bridge = CvBridge()

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self._map_callback, 10)
        self.semantic_sub = self.create_subscription(SemanticObjectArray, self.input_topic, self._semantic_callback, 10)
        self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self._pc_callback, 10)

        # Publishers
        self.publisher = self.create_publisher(ClusteredMapObjectArray, self.output_topic, 10)
        self.vis_pub = self.create_publisher(Image, "/vision/room_segmentation_vis", 10)

        # The Processing Loop
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._process_publish_cycle)

        self.get_logger().info(f"[DEBUG] PointCloud Semantic Masking Node started at {self.publish_rate_hz} Hz.")

    def _map_callback(self, msg: OccupancyGrid):
        self.latest_map_msg = msg

    def _semantic_callback(self, msg: SemanticObjectArray):
        self.latest_semantic_msg = msg

    def _pc_callback(self, msg: PointCloud2):
        self.latest_pc_msg = msg

    def _process_publish_cycle(self):
        # We need all three data streams to do this properly
        if self.latest_map_msg is None or self.latest_semantic_msg is None or self.latest_pc_msg is None:
            return 

        if not self.latest_semantic_msg.objects:
            return 

        # 1. Setup Base Map
        map_info = self.latest_map_msg.info
        width, height = map_info.width, map_info.height
        res = map_info.resolution
        orig_x = map_info.origin.position.x
        orig_y = map_info.origin.position.y

        grid = np.array(self.latest_map_msg.data, dtype=np.int8).reshape((height, width))
        free_space = np.zeros_like(grid, dtype=np.uint8)
        free_space[grid == 0] = 255 

        # 2. Optional POINTCLOUD MASKING (High-Precision Eraser)
        # Toggle with parameter: enable_object_removal
        enable_object_removal = bool(self.get_parameter("enable_object_removal").value)
        if enable_object_removal:
            # We only extract 'x' and 'y' to save processing time
            generator = point_cloud2.read_points(self.latest_pc_msg, field_names=("x", "y"), skip_nans=True)

            # FIX: Force strict unpacking into a list of basic floats so numpy doesn't get confused
            pts_list = [[float(pt[0]), float(pt[1])] for pt in generator]

            if pts_list:
                # Explicitly cast to a 32-bit float matrix. This guarantees a 2D shape (N, 2).
                pts_arr = np.array(pts_list, dtype=np.float32)

                # Vectorized coordinate conversion: Metric -> Pixel
                pxs = ((pts_arr[:, 0] - orig_x) / res).astype(np.int32)
                pys = ((pts_arr[:, 1] - orig_y) / res).astype(np.int32)

                # Vectorized bounds checking to drop points outside the map
                valid_mask = (pxs >= 0) & (pxs < width) & (pys >= 0) & (pys < height)
                pxs = pxs[valid_mask]
                pys = pys[valid_mask]

                # Create a blank mask and stamp the points onto it
                pc_mask = np.zeros_like(free_space, dtype=np.uint8)
                pc_mask[pys, pxs] = 255

                # Dilate the mask. Pointclouds have gaps. Dilation melts the points together
                # into a solid blob to ensure the object is completely erased.
                dilation_px = int(self.get_parameter("pc_dilation_px").value)

                # Using an ellipse kernel prevents unnatural square corners
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
                solid_mask = cv2.dilate(pc_mask, kernel, iterations=1)

                # ERASE: Overwrite the map's free space wherever our solid mask is white
                free_space[solid_mask == 255] = 255

        # 3. Distance Transform & Watershed on the CLEANED map
        dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        thresh_mult = self.get_parameter("dist_thresh_multiplier").value
        _, room_seeds = cv2.threshold(dist_transform, thresh_mult * dist_transform.max(), 255, 0)
        room_seeds = np.uint8(room_seeds)
        
        _, markers = cv2.connectedComponents(room_seeds)
        markers = markers + 1 # Background is 1
        
        unknown = cv2.subtract(free_space, room_seeds)
        markers[unknown == 255] = 0
        img_color = cv2.cvtColor(free_space, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img_color, markers)
        
        self.room_markers = markers
        self._publish_visualization(markers, width, height, map_info)

        # 4. Object Assignment
        self._cluster_and_publish_objects(map_info)

    def _cluster_and_publish_objects(self, map_info):
        """Looks up the room ID for each object and publishes the JSON/ROS Msg."""
        res = map_info.resolution
        orig_x = map_info.origin.position.x
        orig_y = map_info.origin.position.y
        height, width = self.room_markers.shape
        search_radius = self.get_parameter("wall_search_radius_px").value

        labels, coords, obj_ids, names, similarities = [], [], [], [], []

        for obj in self.latest_semantic_msg.objects:
            x, y, z = obj.pose_map.x, obj.pose_map.y, obj.pose_map.z
            px = int((x - orig_x) / res)
            py = int((y - orig_y) / res)
            
            if 0 <= px < width and 0 <= py < height:
                room_id = self._get_nearest_valid_room(px, py, width, height, search_radius)
                labels.append(room_id)
            else:
                labels.append(-1)

            coords.append([x, y, z])
            obj_ids.append(obj.object_id)
            names.append(obj.name)
            similarities.append(obj.similarity)

        # Generate Clusters & Dimensions
        X = np.array(coords)
        labels = np.array(labels)
        unique_labels = sorted(set(labels.tolist()))
        
        centroid_by_label, dimensions_by_label = {}, {}

        for lbl in unique_labels:
            indices = np.where(labels == lbl)[0]
            cluster_points = X[indices]
            centroid = cluster_points.mean(axis=0)

            centroid_by_label[int(lbl)] = {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])}
            min_coords, max_coords = cluster_points.min(axis=0), cluster_points.max(axis=0)
            
            dimensions_by_label[int(lbl)] = {
                "bounding_box": {
                    "min": {"x": float(min_coords[0]), "y": float(min_coords[1])},
                    "max": {"x": float(max_coords[0]), "y": float(max_coords[1])},
                    "dimensions": {"width": float(max_coords[0] - min_coords[0]), "length": float(max_coords[1] - min_coords[1])},
                },
                "radius": float(np.max(np.linalg.norm(cluster_points[:, :2] - centroid[:2], axis=1))),
            }

        final_output = []
        for obj_id, name, coord, label, similarity in zip(obj_ids, names, coords, labels, similarities):
            final_output.append({
                "id": obj_id, "cluster": int(label), "class": name, "similarity": float(similarity),
                "coords": {"x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2])},
                "cluster_centroid": centroid_by_label[int(label)],
                "cluster_dimensions": dimensions_by_label[int(label)],
            })

        if self.output_clustered_map_file:
            self._write_map_to_file(final_output, self.output_clustered_map_file)

        out_msg = self._to_msg(final_output)
        self.publisher.publish(out_msg)

    def _get_nearest_valid_room(self, px: int, py: int, width: int, height: int, max_radius: int) -> int:
        if self.room_markers[py, px] > 1: return self.room_markers[py, px]
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < width and 0 <= ny < height and self.room_markers[ny, nx] > 1:
                            return self.room_markers[ny, nx]
        return 1 

    def _publish_visualization(self, markers, width, height, map_info):
        vis_img = np.zeros((height, width, 3), dtype=np.uint8)
        unique_markers = np.unique(markers)
        for marker in unique_markers:
            if marker == -1: vis_img[markers == marker] = [0, 0, 255] 
            elif marker == 1: vis_img[markers == marker] = [50, 50, 50] 
            else:
                vis_img[markers == marker] = self._room_color(int(marker))

        legend_panel = self._build_legend_panel(height, markers)
        combined = np.hstack((vis_img, legend_panel))

        if self.latest_semantic_msg is not None:
            for obj in self.latest_semantic_msg.objects:
                px_py = self._world_to_pixel(obj.pose_map.x, obj.pose_map.y, map_info)
                if px_py is None:
                    continue

                px, py = px_py
                cv2.circle(combined, (px, py), 4, (0, 0, 0), thickness=-1)
                cv2.circle(combined, (px, py), 2, (255, 255, 255), thickness=-1)

        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(combined, encoding="bgr8"))

    def _build_legend_panel(self, height: int, markers) -> np.ndarray:
        panel_width = 320
        panel = np.full((height, panel_width, 3), 28, dtype=np.uint8)

        cv2.rectangle(panel, (0, 0), (panel_width - 1, height - 1), (90, 90, 90), 1)

        y = 28
        left = 16
        swatch = 18
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(panel, "Legend", (left, y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24

        cv2.putText(panel, "Rooms", (left, y), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        y += 18

        room_markers = [int(marker) for marker in np.unique(markers) if int(marker) > 1]
        room_markers.sort()
        room_markers = [marker for marker in room_markers if marker > 1]

        for marker in room_markers:
            color = self._room_color(marker)
            cv2.rectangle(panel, (left, y - 12), (left + swatch, y + 6), color, thickness=-1)
            cv2.rectangle(panel, (left, y - 12), (left + swatch, y + 6), (0, 0, 0), thickness=1)
            cv2.putText(panel, f"Room {marker}", (left + swatch + 10, y + 2), font, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            y += 22

        if self.latest_semantic_msg is not None and self.latest_semantic_msg.objects:
            y += 8
            cv2.putText(panel, "Objects", (left, y), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            y += 18

            class_counts = Counter(
                str(obj.name).strip() or str(obj.object_id).strip() or "object"
                for obj in self.latest_semantic_msg.objects
            )

            for class_name, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0])):
                if y > height - 18:
                    cv2.putText(panel, "...", (left, height - 14), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
                    break

                cv2.circle(panel, (left + 9, y - 6), 5, (255, 255, 255), thickness=-1)
                cv2.circle(panel, (left + 9, y - 6), 5, (0, 0, 0), thickness=1)
                label = f"{class_name} x{count}" if count > 1 else class_name
                cv2.putText(panel, label, (left + 24, y), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
                y += 20

        cv2.putText(panel, "Map points", (left, height - 18), font, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.circle(panel, (left + 10, height - 32), 4, (255, 255, 255), thickness=-1)
        cv2.circle(panel, (left + 10, height - 32), 4, (0, 0, 0), thickness=1)

        return panel

    @staticmethod
    def _room_color(marker: int) -> tuple:
        np.random.seed(marker)
        return tuple(int(value) for value in np.random.randint(50, 255, size=3))

    @staticmethod
    def _world_to_pixel(x: float, y: float, map_info):
        res = map_info.resolution
        width = map_info.width
        height = map_info.height
        orig_x = map_info.origin.position.x
        orig_y = map_info.origin.position.y

        px = int((x - orig_x) / res)
        py = int((y - orig_y) / res)

        if 0 <= px < width and 0 <= py < height:
            return px, py
        return None

    def _write_map_to_file(self, clustered_map: List[dict], file_path: str) -> None:
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(file_path).open("w", encoding="utf-8") as handle:
                json.dump(clustered_map, handle, indent=4)
        except Exception as exc:
            self.get_logger().error(f"Failed writing json: {exc}")

    @staticmethod
    def _vec3(x: float, y: float, z: float) -> Vector3:
        msg = Vector3()
        msg.x, msg.y, msg.z = float(x), float(y), float(z)
        return msg

    def _to_msg(self, clustered_map: List[dict]) -> ClusteredMapObjectArray:
        out_msg = ClusteredMapObjectArray()
        out_msg.stamp = self.get_clock().now().to_msg()
        out_msg.frame_id = self.frame_id
        for entry in clustered_map:
            obj_msg = ClusteredMapObject()
            obj_msg.id, obj_msg.cluster = str(entry.get("id", "")), int(entry.get("cluster", -1))
            obj_msg.class_name, obj_msg.similarity = str(entry.get("class", "unknown")), float(entry.get("similarity", 0.0))
            
            c, cent = entry.get("coords", {}), entry.get("cluster_centroid", {})
            obj_msg.coords = self._vec3(c.get("x", 0.0), c.get("y", 0.0), c.get("z", 0.0))
            obj_msg.cluster_centroid = self._vec3(cent.get("x", 0.0), cent.get("y", 0.0), cent.get("z", 0.0))

            dims = entry.get("cluster_dimensions", {})
            bbox = dims.get("bounding_box", {})
            bbox_msg = ClusterBoundingBox2D()
            bbox_msg.min_x, bbox_msg.min_y = float(bbox.get("min", {}).get("x", 0.0)), float(bbox.get("min", {}).get("y", 0.0))
            bbox_msg.max_x, bbox_msg.max_y = float(bbox.get("max", {}).get("x", 0.0)), float(bbox.get("max", {}).get("y", 0.0))
            bbox_msg.width, bbox_msg.length = float(bbox.get("dimensions", {}).get("width", 0.0)), float(bbox.get("dimensions", {}).get("length", 0.0))

            dims_msg = ClusterDimensions2D()
            dims_msg.bounding_box, dims_msg.radius = bbox_msg, float(dims.get("radius", 0.0))
            obj_msg.cluster_dimensions = dims_msg
            out_msg.objects.append(obj_msg)
        return out_msg

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusteredMapPreprocPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()