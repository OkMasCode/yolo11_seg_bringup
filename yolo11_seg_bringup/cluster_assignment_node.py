import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import rclpy
import tf2_ros
from tf2_ros import TransformException
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
from yolo11_seg_interfaces.srv import GetApproachPose

class ClusteredMapPreprocPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("clustered_map_preproc_publisher_node")
        self.declare_parameter("output_clustered_map_file", "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/clustered_map_v6.json")
        self.declare_parameter("output_topic", "/vision/clustered_map_v6")
        self.declare_parameter("input_topic", "/vision/semantic_map_v5")
        self.declare_parameter("visualization_topic", "/vision/clustered_map_vis")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 0.5)
        self.declare_parameter("dist_thresh_multiplier", 0.4)
        self.declare_parameter("wall_search_radius_px", 10) 
        self.declare_parameter("min_radius", 0.1)
        self.declare_parameter("max_radius", 2.5)
        self.declare_parameter("radius_step", 0.1)
        self.declare_parameter("angle_step_deg", 5.0)
        self.declare_parameter("free_threshold", 0)
        self.declare_parameter("enable_object_removal", True)
        self.output_clustered_map_file = str(self.get_parameter("output_clustered_map_file").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.vis_topic = str(self.get_parameter("visualization_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.enable_object_removal = bool(self.get_parameter("enable_object_removal").value)
        self.min_radius = float(self.get_parameter("min_radius").value)
        self.max_radius = float(self.get_parameter("max_radius").value)
        self.radius_step = float(self.get_parameter("radius_step").value)
        self.angle_step_deg = float(self.get_parameter("angle_step_deg").value)
        self.free_threshold = int(self.get_parameter("free_threshold").value)
        # State Variables
        self.latest_map_msg = None
        self.latest_costmap_msg = None
        self.latest_semantic_msg = None
        self.local_semantic_map = SemanticObjectArray()
        self.room_markers = None
        self.map_info = None
        self.room_assignments: dict = {}  # object_id → room_id
        self.bridge = CvBridge()
        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, "/jackal/map", self._map_callback, 10)
        self.costmap_sub = self.create_subscription(OccupancyGrid, "/jackal/global_costmap/costmap", self._costmap_callback, 10)
        self.semantic_sub = self.create_subscription(SemanticObjectArray, self.input_topic, self._semantic_callback, 10)
        # Publishers
        self.publisher = self.create_publisher(ClusteredMapObjectArray, self.output_topic, 10)
        self.vis_pub = self.create_publisher(Image, self.vis_topic, 10)
        # Publishing loop (always runs, reads from JSON file)
        self.process_timer = self.create_timer(1.0 / self.publish_rate_hz, self._process_cycle)
        self.publish_timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_cycle)
        # Create the ROS2 Service Server
        self.waypoint_service = self.create_service(
            GetRoomWaypoint, 
            '/vision/get_room_waypoint', 
            self._waypoint_service_callback
        )
        self.approach_service = self.create_service(
            GetApproachPose,
            '/vision/get_approach_pose',
            self._approach_service_callback
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info("[DEBUG] Waypoint Service Server is ready.")
        self.get_logger().info(f"[DEBUG] PointCloud Semantic Masking Node started at {self.publish_rate_hz} Hz.")

    def _map_callback(self, msg: OccupancyGrid):
        self.latest_map_msg = msg

    def _costmap_callback(self, msg: OccupancyGrid):
        self.latest_costmap_msg = msg

    def _semantic_callback(self, msg: SemanticObjectArray):
        self.latest_semantic_msg = msg

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

    def _approach_service_callback(self, request, response):
        if self.latest_map_msg is None or self.room_markers is None or self.map_info is None:
            self.get_logger().error("Cannot compute approach pose: room segmentation or map is not ready.")
            response.success = False
            return response

        goal_pose = request.goal_pose
        start_pose = request.start_pose
        object_room_id = int(request.room_id)
        goal_x = float(goal_pose.pose.position.x)
        goal_y = float(goal_pose.pose.position.y)
        robot_x = float(start_pose.pose.position.x)
        robot_y = float(start_pose.pose.position.y)
        self.get_logger().info(f"BT Node requested approach pose for the goal in Room {object_room_id}")

        min_radius = float(self.min_radius)
        max_radius = float(self.max_radius)
        radius_step = float(self.radius_step)
        angle_step_deg = float(self.angle_step_deg)
        free_threshold = int(self.free_threshold)

        if min_radius < 0.0 or max_radius < min_radius or radius_step <= 0.0 or angle_step_deg <= 0.0:
            self.get_logger().error("Invalid approach pose sampling parameters.")
            response.success = False
            return response

        best_pose = None
        best_robot_distance = float("inf")
        best_goal_distance = float("inf")

        for radius in np.arange(min_radius, max_radius + 1e-6, radius_step):
            for angle_deg in np.arange(0.0, 360.0, angle_step_deg):
                angle_rad = np.deg2rad(angle_deg)
                candidate_x = goal_x + radius * float(np.cos(angle_rad))
                candidate_y = goal_y + radius * float(np.sin(angle_rad))

                if not self._point_in_room(candidate_x, candidate_y, object_room_id):
                    continue
                if not self.isFree(self.latest_costmap_msg, candidate_x, candidate_y, free_threshold):
                    continue

                robot_distance = float(np.hypot(candidate_x - robot_x, candidate_y - robot_y))
                goal_distance = float(np.hypot(candidate_x - goal_x, candidate_y - goal_y))

                if (
                    best_pose is None
                    or robot_distance < best_robot_distance
                    or (
                        abs(robot_distance - best_robot_distance) < 0.01
                        and goal_distance < best_goal_distance
                    )
                ):
                    best_pose = (candidate_x, candidate_y)
                    best_robot_distance = robot_distance
                    best_goal_distance = goal_distance

        if best_pose is None:
            self.get_logger().warn("No valid approach pose found inside the goal room.")
            response.success = False
            return response

        approach_x, approach_y = best_pose
        yaw = float(np.arctan2(goal_y - approach_y, goal_x - approach_x))

        response.success = True
        response.approach_pose.pose.position.x = float(approach_x)
        response.approach_pose.pose.position.y = float(approach_y)
        response.approach_pose.pose.position.z = 0.0
        response.approach_pose.pose.orientation.x = 0.0
        response.approach_pose.pose.orientation.y = 0.0
        response.approach_pose.pose.orientation.z = float(np.sin(yaw * 0.5))
        response.approach_pose.pose.orientation.w = float(np.cos(yaw * 0.5))
        return response

    def _room_id_at_world(self, wx: float, wy: float) -> int:
        """Return the watershed room label at world coordinates, or 1 (background sentinel) if invalid."""
        if self.room_markers is None or self.map_info is None:
            return 1
        res = float(self.map_info.resolution)
        if res <= 0.0:
            return 1
        ox = float(self.map_info.origin.position.x)
        oy = float(self.map_info.origin.position.y)
        px = int((wx - ox) / res)
        py = int((wy - oy) / res)
        h, w = self.room_markers.shape
        if px < 0 or py < 0 or px >= w or py >= h:
            return 1
        return int(self.room_markers[py, px])

    def isFree(self, map_msg: OccupancyGrid, wx: float, wy: float, threshold: int = 0) -> bool:
        """Return True if the occupancy grid cell at world (wx, wy) has value == threshold (0 = free)."""
        info = map_msg.info
        px = int((wx - info.origin.position.x) / info.resolution)
        py = int((wy - info.origin.position.y) / info.resolution)
        if px < 0 or py < 0 or px >= info.width or py >= info.height:
            return False
        return map_msg.data[py * info.width + px] == threshold

    def _point_in_room(self, wx: float, wy: float, room_id: int) -> bool:
        if self.room_markers is None or self.map_info is None:
            return False

        res = float(self.map_info.resolution)
        if res <= 0.0:
            return False

        origin_x = float(self.map_info.origin.position.x)
        origin_y = float(self.map_info.origin.position.y)
        px = int((wx - origin_x) / res)
        py = int((wy - origin_y) / res)

        if px < 0 or py < 0 or py >= self.room_markers.shape[0] or px >= self.room_markers.shape[1]:
            return False

        return int(self.room_markers[py, px]) == int(room_id)

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

    def _process_cycle(self):
        if self.latest_map_msg is None:
            return
        # Room segmentation only needs the map — compute it regardless of semantic messages
        map_info = self.latest_map_msg.info
        self.map_info = map_info
        width, height = map_info.width, map_info.height
        grid = np.array(self.latest_map_msg.data, dtype=np.int8).reshape((height, width))
        free_space = np.zeros_like(grid, dtype=np.uint8)
        free_space[grid == 0] = 255
        dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        thresh_mult = self.get_parameter("dist_thresh_multiplier").value
        _, room_seeds = cv2.threshold(dist_transform, thresh_mult * dist_transform.max(), 255, 0)
        room_seeds = np.uint8(room_seeds)
        _, markers = cv2.connectedComponents(room_seeds)
        markers = markers + 1
        unknown = cv2.subtract(free_space, room_seeds)
        markers[unknown == 255] = 0
        img_color = cv2.cvtColor(free_space, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img_color, markers)
        self.room_markers = markers
        self._publish_visualization(markers, width, height, map_info)
        # Cluster assignment requires semantic messages — skip if mapper is not running
        if self.latest_semantic_msg is None or not self.latest_semantic_msg.objects:
            return
        self._cluster_and_publish_objects(map_info)

    def _cluster_and_publish_objects(self, map_info):
        """Assigns room IDs to new objects via robot pose at detection time, then publishes all accumulated objects."""
        # --- Process only NEW objects ---
        existing_ids = {o.object_id for o in self.local_semantic_map.objects}
        for obj in self.latest_semantic_msg.objects:
            if obj.object_id in existing_ids:
                continue
            room_id = self._get_current_room_id(obj.timestamp)
            self.room_assignments[obj.object_id] = room_id
            self.local_semantic_map.objects.append(obj)

        # --- Build output from ALL accumulated objects ---
        if not self.local_semantic_map.objects:
            return

        labels, coords, obj_ids, names, similarities = [], [], [], [], []
        for obj in self.local_semantic_map.objects:
            obj_ids.append(obj.object_id)
            names.append(obj.name)
            coords.append([obj.pose_map.x, obj.pose_map.y, obj.pose_map.z])
            labels.append(self.room_assignments.get(obj.object_id, -1))
            similarities.append(obj.similarity)

        labels = np.array(labels)
        unique_labels = sorted(set(labels.tolist()))
        centroid_by_label = {}
        for lbl in unique_labels:
            centroid = self.generate_exploration_waypoints(target_room_id=lbl, num_waypoints=1)
            if not centroid:
                self.get_logger().error(f"Failed to generate a safe point for Room {lbl}!")
                centroid_by_label[int(lbl)] = {"x": 0.0, "y": 0.0, "z": 0.0}
            else:
                centroid_by_label[int(lbl)] = {
                    "x": float(centroid[0][0]),
                    "y": float(centroid[0][1]),
                    "z": 0.0
                }
        final_output = []
        for obj_id, name, coord, label, similarity in zip(obj_ids, names, coords, labels, similarities):
            final_output.append({
                "id": obj_id, "cluster": int(label), "class": name, "similarity": float(similarity),
                "coords": {"x": float(coord[0]), "y": float(coord[1])},
                "cluster_centroid": centroid_by_label[int(label)],
            })
        if self.output_clustered_map_file:
            self._write_map_to_file(final_output, self.output_clustered_map_file)

    def _publish_cycle(self):
        """Reads the clustered map JSON and publishes it. Runs independently of semantic messages."""
        if not self.output_clustered_map_file:
            return
        path = Path(self.output_clustered_map_file)
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                clustered_map = json.load(f)
        except Exception as exc:
            self.get_logger().error(f"Failed reading clustered map json: {exc}")
            return
        out_msg = self._to_msg(clustered_map)
        self.publisher.publish(out_msg)

    def _get_current_room_id(self, timestamp) -> int:
        try:
            t = self.tf_buffer.lookup_transform('map','base_link', timestamp)
            robot_x = t.transform.translation.x
            robot_y = t.transform.translation.y
            robot_room_id = self._room_id_at_world(robot_x, robot_y)
            return robot_room_id
        except TransformException as e:
            self.get_logger().warn(f'Could not get robot pose: {e}')
            return -1   

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
            #bbox_msg.min_x, bbox_msg.min_y = float(bbox.get("min", {}).get("x", 0.0)), float(bbox.get("min", {}).get("y", 0.0))
            #bbox_msg.max_x, bbox_msg.max_y = float(bbox.get("max", {}).get("x", 0.0)), float(bbox.get("max", {}).get("y", 0.0))
            bbox_msg.min_x, bbox_msg.min_y = 0.0, 0.0
            bbox_msg.max_x, bbox_msg.max_y = 0.0, 0.0
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