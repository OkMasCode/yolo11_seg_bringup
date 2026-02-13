from polars import Duration
from tf2_ros import Buffer, TransformException
from geometry_msgs.msg import TransformStamped, Vector3
from rclpy.node import Node
import rclpy
from rclpy.duration import Duration

from collections import namedtuple
import os
from builtin_interfaces.msg import Time
import numpy as np
import math
import json
from typing import Tuple

_EPS = np.finfo(float).eps * 4.0 # Small epsilon value to avoid division by zero in quaternion normalization

ObjectEntry = namedtuple('ObjectEntry', ['frame', 'timestamp', 'pose_cam', 'pose_map', 'occurrences', 'name', 'similarity', 'image_embedding', 'confidence', 'box_size'])

# ----------------- FUNCTIONS ------------------ #

def quaternion_matrix(quaternion):
    """
    Convert a quaternion into a 4x4 transformation matrix.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

# -------------------- CLASS ------------------- #

class SemanticObjectMap:
    """
    Class to handle detections:
    - Transform detections to map frame
    - Merge close detections
    - Export map to JSON
    """

    def __init__(self, tf_buffer: Buffer, node: Node):
        self.objects = {}
        self.tf_buffer = tf_buffer
        self.node = node
    
    def add_detection(self, object_name: str, object_id: str, pose_in_camera, detection_stamp, camera_frame='camera3_color_optical_frame', fixed_frame='camera3_color_optical_frame', distance_threshold=0.8, embeddings=None, goal_embedding=None, similarity=0.0, confidence=0.0, box_min=None, box_max=None):
        """
        Add a new detection to the semantic map.
        Transforms the pose to the fixed frame and merges with existing objects if close enough.
        """
        try:
            lookup_time = rclpy.time.Time.from_msg(detection_stamp)
            # Get the transform from camera frame to fixed frame
            transform = self.tf_buffer.lookup_transform(
                fixed_frame,
                camera_frame,
                lookup_time,
                timeout=Duration(seconds=0.8)
            )
            # Transform the pose to the fixed frame
            pose_in_map = self.transform_point(pose_in_camera, transform)
            
            self.node.get_logger().info(f"Centroid in map frame for {object_name}: x={pose_in_map[0]:.3f}, y={pose_in_map[1]:.3f}, z={pose_in_map[2]:.3f}")

            # Prepare embeddings; keep existing if incoming is empty/None
            img_vec = None
            if embeddings is not None:
                emb_arr = np.asarray(embeddings, dtype=np.float32)
                if emb_arr.size > 0:
                    img_vec = emb_arr

            new_size = (
                    abs(box_max[0] - box_min[0]),
                    abs(box_max[1] - box_min[1]),
                    abs(box_max[2] - box_min[2])
                )
            # Iterate over stored objects and check if this detection is close to any.
            for existing_id, entry in self.objects.items():
                dist = self.euclidean_distance(pose_in_map, entry.pose_map)

                overlap = False
                if entry.box_size is not None:
                    overlap = self.check_aabb_intersection(
                        center_a=pose_in_map,   size_a=new_size,
                        center_b=entry.pose_map, size_b=entry.box_size
                    )
                if dist < distance_threshold or overlap:
                    avg_pose = tuple(
                        (entry.pose_map[i] * entry.occurrences + pose_in_map[i]) / (entry.occurrences + 1)
                        for i in range(3)
                    )
                    new_similarity = (entry.similarity + similarity) / 2
                    # Update confidence only if new one is higher
                    updated_confidence = max(entry.confidence, confidence)
                    updated_embedding = entry.image_embedding if img_vec is None else img_vec
                    self.objects[existing_id] = entry._replace(
                        pose_map=avg_pose,
                        occurrences=entry.occurrences + 1,
                        similarity=new_similarity,
                        image_embedding=updated_embedding,
                        confidence=updated_confidence,
                    )
                    return False

            self.objects[object_id] = ObjectEntry(
                frame = camera_frame,
                timestamp = detection_stamp,
                pose_cam = (pose_in_camera.x, pose_in_camera.y, pose_in_camera.z),
                pose_map = pose_in_map,
                occurrences = 1,
                name = object_name,
                similarity = similarity,
                image_embedding = img_vec,
                confidence= confidence,
                box_size= new_size
            )
            return True

        except TransformException as ex:
            self.node.get_logger().warning(f"Could not store {object_id}: {ex}")
            return False
        
    def transform_point(self, point, transform: TransformStamped):
        """
        Apply the transform (rotation + translation) to a Vector3 point.

        Args:
            point: geometry_msgs.msg.Vector3 representing coordinates in the source frame
            transform: TransformStamped that maps the source frame into some target frame

        Returns:
            A tuple (x, y, z) representing the transformed point in the target frame.
        """
        q = transform.transform.rotation # rotation of the camera frame with respect to the absolute frame
        t = transform.transform.translation # translation of the camera frame with respect to the absolute frame

        rot_matrix = quaternion_matrix([q.w, q.x, q.y, q.z])[:3, :3] # obtains the rotation matrix from the quaternion values

        point_vec = np.array([[point.x], [point.y], [point.z]]) # this is the coordinates of the object in camera frame
        translation = np.array([[t.x], [t.y], [t.z]]) # this is the coordinates of the camera in abosulte frame

        # result = R * point + t
        result = rot_matrix @ point_vec + translation 
        # Return as a simple (x, y, z) tuple
        return (result[0, 0], result[1, 0], result[2, 0])

    def euclidean_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Compute Euclidean distance between two 3D points represented as (x,y,z)."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    
    def update_all_to_latest_map(self, fixed_frame: str = 'odom') -> None:
        """
        Recompute pose_map for every stored object using the transform available in tf.

        This method attempts to re-transform each object's stored camera-frame pose
        into the fixed_frame using the transform at the object's stored timestamp.
        """
        for object_id, entry in self.objects.items():
            try:
                # Lookup the transform from the object's stored frame (entry.frame) to the fixed_frame
                # at the time stored in entry.timestamp.
                transform = self.tf_buffer.lookup_transform(
                    fixed_frame,
                    entry.frame,
                    rclpy.time.Time.from_msg(entry.timestamp)
                )

                # Re-transform using the original camera-frame position saved in entry.pose_cam.
                # Construct a Vector3 from the stored tuple to pass to transform_point.
                point = Vector3(x=entry.pose_cam[0], y=entry.pose_cam[1], z=entry.pose_cam[2])
                new_pose = self.transform_point(point, transform)
                # Update the stored entry's pose_map with the recomputed pose (preserve embeddings)
                self.objects[object_id] = entry._replace(pose_map=new_pose)

            except TransformException as ex:
                self.node.get_logger().warning(f"Could not update {object_id}: {ex}")

    def update_similarities(self, goal_embedding) -> None:
        """
        Update similarity scores for all stored objects based on a new goal embedding.
        Args:
            goal_embedding: The new target embedding (should be normalized)
        """
        txt_vec = np.array(goal_embedding, dtype=np.float32)
        
        for object_id, entry in self.objects.items():
            if entry.image_embedding is not None:
                similarity = float(np.dot(entry.image_embedding, txt_vec))
                self.objects[object_id] = entry._replace(similarity=similarity)
            else:
                self.node.get_logger().warning(
                    f"No image embedding stored for {object_id}, cannot update similarity."
                )

    def load_from_json(self, directory_path, file='map.json'):
        """
        Load semantic object map from a JSON file.
        """
        path = os.path.join(directory_path, file)
        
        if not os.path.exists(path):
            self.node.get_logger().info(f"No existing map file found at {path}, starting with empty map")
            return
        
        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
            
            for object_id, obj_data in data.items():
                # Reconstruct the ObjectEntry
                timestamp = Time(
                    sec=obj_data['timestamp']['sec'],
                    nanosec=obj_data['timestamp']['nanosec']
                )
                
                pose_map = (
                    obj_data['pose_map']['x'],
                    obj_data['pose_map']['y'],
                    obj_data['pose_map']['z']
                )
                
                # Load image embedding if available
                image_embedding = np.array(obj_data['image_embedding'], dtype=np.float32) if obj_data.get('image_embedding') else None
                
                self.objects[object_id] = ObjectEntry(
                    frame=obj_data['frame'],
                    timestamp=timestamp,
                    pose_cam=pose_map,  # Use map pose since original camera pose may not be available
                    pose_map=pose_map,
                    occurrences=obj_data['occurrences'],
                    name=obj_data['name'],
                    similarity=obj_data.get('similarity', 0.0),
                    image_embedding=image_embedding,
                    confidence=obj_data.get('confidence', 0.0)

                )
            
            self.node.get_logger().info(f"Loaded {len(self.objects)} objects from {path}")
            
        except Exception as e:
            self.node.get_logger().error(f"Error loading map from {path}: {e}")
                
    def export_to_json(self, directory_path, file='map.json'):
        """
        Export the semantic object map to a JSON file.
        """

        os.makedirs(directory_path, exist_ok=True)
        path = os.path.join(directory_path, file)


        export_data = {}
        for object_id, entry in self.objects.items():
            # Convert similarity to native Python float, handling None values
            similarity_value = float(entry.similarity) if entry.similarity is not None else None
            
            export_data[object_id] = {
                'name': entry.name,
                'frame': entry.frame,
                'timestamp': {
                    'sec': entry.timestamp.sec,
                    'nanosec': entry.timestamp.nanosec
                },
                'pose_map': {
                    'x': float(entry.pose_map[0]),
                    'y': float(entry.pose_map[1]),
                    'z': float(entry.pose_map[2])
                },
                'occurrences': int(entry.occurrences),
                'similarity': similarity_value,
                'image_embedding': entry.image_embedding.tolist() if entry.image_embedding is not None else None,
                'confidence': float(entry.confidence) if entry.confidence is not None else 0.0
            }

        with open(path, 'w') as json_file:
            json.dump(export_data, json_file, indent=4)
        return
    
    def check_aabb_intersection(self, center_a, size_a, center_b, size_b):
        """
        Manual AABB Intersection check.
        center: (x, y, z)
        size: (width, height, depth) aka (dx, dy, dz)
        """
        # Calculate half-sizes
        ha = np.array(size_a) / 2.0
        hb = np.array(size_b) / 2.0
        
        # Calculate min/max for A
        min_a = np.array(center_a) - ha
        max_a = np.array(center_a) + ha
        
        # Calculate min/max for B
        min_b = np.array(center_b) - hb
        max_b = np.array(center_b) + hb
        
        # Check overlap in all 3 axes
        overlap_x = (min_a[0] <= max_b[0]) and (max_a[0] >= min_b[0])
        overlap_y = (min_a[1] <= max_b[1]) and (max_a[1] >= min_b[1])
        overlap_z = (min_a[2] <= max_b[2]) and (max_a[2] >= min_b[2])
        
        return overlap_x and overlap_y and overlap_z