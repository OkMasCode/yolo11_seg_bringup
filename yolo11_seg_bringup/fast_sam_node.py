#!/usr/bin/env python3

"""
ROS2 FastSAM Tracking Node
Subscribes to an RGB camera feed, runs FastSAM class-agnostic segmentation 
with object tracking, and publishes the annotated output.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge

import cv2
from ultralytics import FastSAM


class FastSAMNode(Node):
    """
    ROS2 Node that handles class-agnostic segmentation and tracking using FastSAM.
    """
    def __init__(self):
        super().__init__('fast_sam_node')
        self.get_logger().info("Initializing FastSAM Tracking Node...")

        # ============= Parameters ============= #
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('output_topic', '/vision/annotated_image')
        # FastSAM-s (small) is highly recommended for edge devices like Jetson
        self.declare_parameter('model_path', 'FastSAM-s.pt') 
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('tracker', 'botsort.yaml') # 'botsort.yaml' or 'bytetrack.yaml'

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.conf = self.get_parameter('conf').value
        self.iou = self.get_parameter('iou').value
        self.tracker_type = self.get_parameter('tracker').value

        # ============= Model Initialization ============= #
        self.get_logger().info(f"Loading FastSAM model: {self.model_path}")
        try:
            self.model = FastSAM(self.model_path)
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model. Ensure the path is correct: {e}")
            raise SystemExit

        # Utilities
        self.bridge = CvBridge()

        # QoS profile optimized for sensor data (drop old frames if lagging)
        qos_sensor = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # ============= Publishers & Subscribers ============= #
        self.sub = self.create_subscription(
            Image, 
            self.input_topic, 
            self.image_callback, 
            qos_profile=qos_sensor
        )
        self.pub = self.create_publisher(Image, self.output_topic, 10)
        
        self.get_logger().info("FastSAM Node is ready and listening to images.")

    def _filter_contained_masks(self, masks, boxes, ioa_threshold=0.85):
        """
        Identifies and removes sub-masks (like a beard) that are physically 
        contained inside larger parent masks (like a person).
        """
        import numpy as np
        N = len(masks)
        keep_indices = np.ones(N, dtype=bool)
        
        # Calculate pixel areas of all masks
        areas = masks.sum(axis=(1, 2))
        
        for i in range(N):
            if not keep_indices[i]: continue
                
            for j in range(N):
                if i == j or not keep_indices[j]: continue
                    
                # Quick bounding box overlap check (x1, y1, x2, y2)
                box1, box2 = boxes[i], boxes[j]
                overlap = not (box1[2] < box2[0] or box1[0] > box2[2] or 
                               box1[3] < box2[1] or box1[1] > box2[3])
                
                if not overlap:
                    continue
                
                # If Mask i is smaller than Mask j, check if j swallows i
                if areas[i] < areas[j]:
                    # Bitwise AND to find intersecting pixels
                    intersection = np.logical_and(masks[i], masks[j]).sum()
                    ioa = intersection / areas[i]
                    
                    # If the smaller mask is >85% inside the larger mask, drop it
                    if ioa > ioa_threshold:
                        keep_indices[i] = False
                        break 
                        
        return keep_indices

    def image_callback(self, msg: Image):
        import numpy as np 
        import cv2
        
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_h, img_w = cv_bgr.shape[:2]

            results = self.model.track(
                source=cv_bgr,
                conf=self.conf,
                iou=self.iou,
                persist=True,
                tracker=self.tracker_type, 
                retina_masks=False,
                verbose=False      
            )

            res = results[0]
            annotated_frame = cv_bgr.copy()

            if res.boxes is not None and hasattr(res, 'masks') and res.masks is not None and res.boxes.id is not None:
                
                # 1. Extract data from GPU to CPU
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                masks = res.masks.data.cpu().numpy() 
                
                # ==========================================
                # NEW: Apply the Hierarchical Containment Filter
                # ==========================================
                keep_indices = self._filter_contained_masks(masks, boxes, ioa_threshold=0.85)
                
                for i in range(len(ids)):
                    # Check if this mask was flagged as a sub-part
                    if not keep_indices[i]:
                        continue # Skip the beard, only draw the person
                    
                    box = boxes[i]
                    track_id = ids[i]
                    mask = masks[i]
                    
                    # 2. FILTERING: By Bounding Box Area
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area < 1500 or area > (img_w * img_h * 0.70):
                        continue
                        
                    # 3. COLORING: Consistent color by ID
                    np.random.seed(track_id)
                    color = np.random.randint(50, 255, size=(3,)).tolist()
                    
                    mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    colored_mask = np.zeros_like(annotated_frame)
                    colored_mask[mask_resized > 0.5] = color
                    
                    alpha = 0.5 
                    mask_indices = mask_resized > 0.5
                    annotated_frame[mask_indices] = cv2.addWeighted(
                        annotated_frame, 1 - alpha, 
                        colored_mask, alpha, 0
                    )[mask_indices]
                    
                    # 4. LABELING
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    cv2.putText(annotated_frame, f"ID: {track_id}", (cx-20, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(annotated_frame, f"ID: {track_id}", (cx-20, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            vis_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            vis_msg.header = msg.header 
            self.pub.publish(vis_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FastSAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down FastSAM Node...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()