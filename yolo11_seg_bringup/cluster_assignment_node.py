import json
from pathlib import Path
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

from yolo11_seg_interfaces.msg import (
    ClusterBoundingBox2D,
    ClusteredMapObject,
    ClusteredMapObjectArray,
    SemanticObjectArray
)

from yolo11_seg_interfaces.srv import GetRoomWaypoint

class ClusteredMapPreprocPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("clustered_map_preproc_publisher_node")
        self.declare_parameter("output_clustered_map_file", "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/clustered_map_v6.json")
        self.declare_parameter("output_topic", "/vision/clustered_map_v6")
        self.declare_parameter("input_topic", "/vision/semantic_map_v5")
        self.declare_parameter("pointcloud_topic", "/vision/semantic_map_v5/points")
        self.declare_parameter("visualization_topic", "/vision/clustered_map_vis")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 0.5)
        # --- TUNING PARAMETERS ---
        self.declare_parameter("dist_thresh_multiplier", 0.4)
        self.declare_parameter("wall_search_radius_px", 10) 
        # Dilation kernel size. Since pointclouds have gaps between points, 
        # we dilate the footprint by this many pixels to create a solid eraser stamp.
        # 5 pixels * 0.05m = ~25cm dilation.
        self.declare_parameter("pc_dilation_px", 8) 
        self.declare_parameter("enable_object_removal", True)
        self.output_clustered_map_file = str(self.get_parameter("output_clustered_map_file").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.pc_topic = str(self.get_parameter("pointcloud_topic").value)
        self.vis_topic = str(self.get_parameter("visualization_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.enable_object_removal = bool(self.get_parameter("enable_object_removal").value)

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
        self.vis_pub = self.create_publisher(Image, self.vis_topic, 10)
        # The Processing Loop
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._process_publish_cycle)
        # Create the ROS2 Service Server
        self.waypoint_service = self.create_service(
            GetRoomWaypoint, 
            '/vision/get_room_waypoint', 
            self._waypoint_service_callback
        )
        self.get_logger().info("[DEBUG] Waypoint Service Server is ready.")
        self.get_logger().info(f"[DEBUG] PointCloud Semantic Masking Node started at {self.publish_rate_hz} Hz.")

    def _map_callback(self, msg: OccupancyGrid):
        self.latest_map_msg = msg

    def _semantic_callback(self, msg: SemanticObjectArray):
        self.latest_semantic_msg = msg

    def _pc_callback(self, msg: PointCloud2):
        self.latest_pc_msg = msg

    def _waypoint_service_callback(self, request, response):
        """Called whenever the C++ BT Node requests a safe point."""
        room_id = request.room_id
        self.get_logger().info(f"BT Node requested waypoint for Room {room_id}")
        # Use the erosion logic to get 1 safe point
        safe_points = self.generate_exploration_waypoints(target_room_id=room_id, num_waypoints=1)
        if not safe_points:
            self.get_logger().error(f"Failed to generate a safe point for Room {room_id}!")
            response.success = False
            return response
        # Pack the response
        response.success = True
        response.waypoint.x = float(safe_points[0][0])
        response.waypoint.y = float(safe_points[0][1])
        response.waypoint.z = 0.0
        return response

    def generate_exploration_waypoints(self, target_room_id: int, num_waypoints: int = 5) -> list:
        """
        Generates safe, random (x,y) metric waypoints strictly inside a specific room.
        """
        if self.room_markers is None or self.map_info is None:
            self.get_logger().error("Cannot generate waypoints: Map not segmented yet.")
            return []
        # Create a binary mask of ONLY the target room
        # 255 = Inside the room, 0 = Outside the room
        room_mask = np.zeros_like(self.room_markers, dtype=np.uint8)
        room_mask[self.room_markers == target_room_id] = 255
        # Sanity check: Does the room exist?
        if np.count_nonzero(room_mask) == 0:
            self.get_logger().error(f"Room {target_room_id} not found in map.")
            return []
        # Erode the mask by the Robot's Radius (e.g., 35cm)
        # This prevents generating waypoints right against the wall where the robot would crash
        ROBOT_RADIUS_M = 0.35
        res = self.map_info.resolution
        erosion_px = int(ROBOT_RADIUS_M / res)  # Convert meters to pixels     
        # Ensure kernel size is odd
        kernel_size = erosion_px if erosion_px % 2 != 0 else erosion_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        safe_room_mask = cv2.erode(room_mask, kernel, iterations=1)
        # Fallback: If the room is so small that eroding it wipes it out completely,
        # fallback to the uneroded mask so we at least get *some* point.
        if np.count_nonzero(safe_room_mask) == 0:
            self.get_logger().warn(f"Room {target_room_id} is too narrow for safe erosion. Using raw boundaries.")
            safe_room_mask = room_mask
        # Extract all valid (row, col) pixel coordinates from the safe mask
        valid_pixels = np.column_stack(np.where(safe_room_mask == 255))
        # Randomly sample N pixels
        # If the user asks for more waypoints than there are pixels, just take all available pixels
        sample_size = min(num_waypoints, len(valid_pixels))
        # np.random.choice requires a 1D array, so we shuffle and slice manually for 2D
        np.random.shuffle(valid_pixels)
        chosen_pixels = valid_pixels[:sample_size]
        # Convert Pixel coordinates back to Metric (x, y) map coordinates
        orig_x = self.map_info.origin.position.x
        orig_y = self.map_info.origin.position.y
        waypoints_metric = []
        for py, px in chosen_pixels:
            metric_x = (px * res) + orig_x
            metric_y = (py * res) + orig_y
            waypoints_metric.append((metric_x, metric_y))
        return waypoints_metric

    def _process_publish_cycle(self):
        # We need all three data streams to do this properly
        if self.latest_map_msg is None or self.latest_semantic_msg is None or self.latest_pc_msg is None:
            return 
        if not self.latest_semantic_msg.objects:
            return 
        # Setup Base Map
        map_info = self.latest_map_msg.info
        width, height = map_info.width, map_info.height
        res = map_info.resolution
        orig_x = map_info.origin.position.x
        orig_y = map_info.origin.position.y
        grid = np.array(self.latest_map_msg.data, dtype=np.int8).reshape((height, width))
        free_space = np.zeros_like(grid, dtype=np.uint8)
        free_space[grid == 0] = 255 
        # Optional POINTCLOUD MASKING (High-Precision Eraser)
        # Toggle with parameter: enable_object_removal
        if self.enable_object_removal:
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
        # Distance Transform & Watershed on the CLEANED map
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
        # Object Assignment
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
        # Generate Clusters
        X = np.array(coords)
        labels = np.array(labels)
        unique_labels = sorted(set(labels.tolist()))
        centroid_by_label = {}
        for lbl in unique_labels:
            indices = np.where(labels == lbl)[0]
            centroid = self.generate_exploration_waypoints(target_room_id=lbl, num_waypoints=1)
            if not centroid:
                self.get_logger().error(f"Failed to generate a safe point for Room {lbl}!")
            centroid_by_label[int(lbl)] = {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])}
        final_output = []
        for obj_id, name, coord, label, similarity in zip(obj_ids, names, coords, labels, similarities):
            final_output.append({
                "id": obj_id, "cluster": int(label), "class": name, "similarity": float(similarity),
                "coords": {"x": float(coord[0]), "y": float(coord[1])},
                "cluster_centroid": centroid_by_label[int(label)],
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
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8"))

    @staticmethod
    def _room_color(marker: int) -> tuple:
        np.random.seed(marker)
        return tuple(int(value) for value in np.random.randint(50, 255, size=3))

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
            bbox_msg = ClusterBoundingBox2D()
            bbox_msg.min_x, bbox_msg.min_y = float(bbox.get("min", {}).get("x", 0.0)), float(bbox.get("min", {}).get("y", 0.0))
            bbox_msg.max_x, bbox_msg.max_y = float(bbox.get("max", {}).get("x", 0.0)), float(bbox.get("max", {}).get("y", 0.0))
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