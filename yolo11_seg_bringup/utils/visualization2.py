#!/usr/bin/env python3
"""Visualization utilities for debugging."""
import cv2
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker, MarkerArray


class Visualizer:
    """Handles visualization tasks for debugging."""
    
    @staticmethod
    def get_color_for_class(class_id: str, class_colors: dict):
        """
        Deterministically map a class_id string to an RGB color.
        
        Args:
            class_id: Class identifier (as string)
            class_colors: Dictionary cache for colors
            
        Returns:
            tuple: (r, g, b) color values
        """
        if class_id not in class_colors:
            h = abs(hash(class_id))
            r = (h >> 0) & 0xFF
            g = (h >> 8) & 0xFF
            b = (h >> 16) & 0xFF
            if r < 30 and g < 30 and b < 30:
                r = (r + 128) & 0xFF
                g = (g + 64) & 0xFF
            class_colors[class_id] = (r, g, b)
        return class_colors[class_id]
    
    @staticmethod
    def draw_clip_boxes(image, sx1, sy1, sx2, sy2, x1, y1, x2, y2, instance_id, class_id):
        """
        Draw CLIP square crop and original bounding box on image.
        
        Args:
            image: BGR image to draw on
            sx1, sy1, sx2, sy2: Square crop coordinates
            x1, y1, x2, y2: Original bbox coordinates
            instance_id: Instance identifier
            class_id: Class identifier
        """
        # Draw CLIP square box (green)
        cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
        # Draw original bbox (red)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # Add label
        label = f"ID:{instance_id} cls:{class_id}"
        cv2.putText(image, label, (sx1, sy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    @staticmethod
    def create_centroid_marker(class_name: str, centroid: Vector3, class_id: int, 
                              marker_id: int, frame_id: str, stamp, color_rgb):
        """
        Create a marker for centroid visualization.
        
        Args:
            class_name: Object class name
            centroid: 3D centroid position
            class_id: Class identifier
            marker_id: Unique marker ID
            frame_id: Reference frame
            stamp: ROS timestamp
            color_rgb: Tuple (r, g, b) in 0-255 range
            
        Returns:
            Marker message
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "yolo_centroids"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(centroid.x)
        marker.pose.position.y = float(centroid.y)
        marker.pose.position.z = float(centroid.z)
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        r, g, b = color_rgb
        marker.color.r = float(r) / 255.0
        marker.color.g = float(g) / 255.0
        marker.color.b = float(b) / 255.0
        marker.color.a = 0.9
        
        return marker
