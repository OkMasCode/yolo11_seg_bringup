#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RoomSegmenter(Node):
    def __init__(self):
        super().__init__('room_segmenter')
        
        # We use a latched-style QoS for the map, assuming it doesn't update at 30fps
        # If your map is static, standard QoS is fine, but we'll use a standard subscription here.
        self.subscription = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            10
        )
        
        # Publish the result as an Image for easy RViz2 viewing
        self.image_pub = self.create_publisher(Image, '/room_segmentation/vis', 10)
        self.bridge = CvBridge()
        
        # Declare a ROS parameter so you can tune the threshold live via CLI
        # Example: ros2 param set /room_segmenter dist_thresh_multiplier 0.6
        self.declare_parameter('dist_thresh_multiplier', 0.4)
        
        # Cache for object lookup later
        self.latest_markers = None
        self.map_info = None
        
        self.get_logger().info("Room Segmenter Node Started. Waiting for /map...")

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        self.map_info = msg.info
        
        # 1. Convert ROS OccupancyGrid to 2D Numpy Array
        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        
        # 2. Create Binary Mask (Free space = 255, Walls/Unknown = 0)
        free_space = np.zeros_like(grid, dtype=np.uint8)
        free_space[grid == 0] = 255 
        
        # 3. Distance Transform
        dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        
        # 4. Apply Live Threshold
        thresh_mult = self.get_parameter('dist_thresh_multiplier').value
        _, room_seeds = cv2.threshold(dist_transform, thresh_mult * dist_transform.max(), 255, 0)
        room_seeds = np.uint8(room_seeds)
        
        # 5. Connected Components (Labeling the seeds)
        _, markers = cv2.connectedComponents(room_seeds)
        
        # Shift markers by 1 so background is 1, not 0 (Watershed requirement)
        markers = markers + 1
        
        # Mark unknown regions with 0
        unknown = cv2.subtract(free_space, room_seeds)
        markers[unknown == 255] = 0
        
        # 6. Watershed Algorithm
        img_color = cv2.cvtColor(free_space, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img_color, markers)
        
        # Store for your object detection lookup pipeline
        self.latest_markers = markers
        
        # 7. Visualization generation for RViz2
        self.publish_visualization(markers, width, height)

    def publish_visualization(self, markers, width, height):
        # Create a blank BGR image
        vis_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        unique_markers = np.unique(markers)
        
        for marker in unique_markers:
            if marker == -1:
                # Boundaries drawn by watershed -> RED
                vis_img[markers == marker] = [0, 0, 255] 
            elif marker == 1:
                # Background / Walls -> DARK GRAY
                vis_img[markers == marker] = [50, 50, 50] 
            else:
                # Rooms -> Assign a consistent random color based on ID
                np.random.seed(marker)
                color = np.random.randint(50, 255, size=3).tolist()
                vis_img[markers == marker] = color
                
        # Convert OpenCV image to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
        
        # Publish
        self.image_pub.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RoomSegmenter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()