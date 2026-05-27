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
        self.declare_parameter("visualization_topic", "/vision/clustered_map_vis")
        self.declare_parameter("publish_rate_hz", 0.5)
        self.declare_parameter("dist_thresh_multiplier", 0.15)
        self.declare_parameter("wall_search_radius_px", 10) 
        self.vis_topic = str(self.get_parameter("visualization_topic").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.latest_map_msg = None
        self.room_markers = None
        self.bridge = CvBridge()
        self.map_sub = self.create_subscription(OccupancyGrid, "/jackal/map", self._map_callback, 10)
        self.vis_pub = self.create_publisher(Image, self.vis_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._process_publish_cycle)
        self.map_info = None

    def _map_callback(self, msg: OccupancyGrid):
        self.latest_map_msg = msg

    def _process_publish_cycle(self):
        # We need all three data streams to do this properly
        if self.latest_map_msg is None:
            return 
        self.map_info = self.latest_map_msg.info
        width, height = self.map_info.width, self.map_info.height
        res = self.map_info.resolution
        orig_x = self.map_info.origin.position.x
        orig_y = self.map_info.origin.position.y
        grid = np.array(self.latest_map_msg.data, dtype=np.int8).reshape((height, width))
        free_space = np.zeros_like(grid, dtype=np.uint8)
        free_space[grid == 0] = 255 
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
        self._publish_visualization(markers, width, height, self.map_info)

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