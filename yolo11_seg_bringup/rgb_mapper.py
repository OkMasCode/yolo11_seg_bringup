from tf2_ros import Buffer, TransformException
from geometry_msgs.msg import TransformStamped, Vector3
from rclpy.node import Node
import rclpy

from collections import namedtuple
import os
from builtin_interfaces.msg import Time
import numpy as np
import math
import json
from typing import Tuple

_EPS = np.finfo(float).eps * 4.0 # Small epsilon value to avoid division by zero in quaternion normalization

ObjectEntry = namedtuple('ObjectEntry', ['frame', 'timestamp', 'pose_cam', 'pose_map', 'occurrences', 'name', 'similarity', 'image_embedding'])

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

    def __init__(self, tf_buffer: Buffer, node: Node, use_pixel_frame: bool = True):
        self.objects = {}
        self.tf_buffer = tf_buffer
        self.node = node
        # When True, centroids are treated as pixel coordinates (RGB case)
        # and no TF transform is applied.
        self.use_pixel_frame = use_pixel_frame
    
    def add_detection(self, object_name: str, object_id: str, pose_in_camera, detection_stamp, camera_frame='camera3_color_optical_frame', fixed_frame='camera3_color_optical_frame', distance_threshold=120, embeddings=None, goal_embedding=None):
        """
        Add a new detection to the semantic map.
        Transforms the pose to the fixed frame and merges with existing objects if close enough.
        """
        try:
            # In RGB mode, pose is in pixel coordinates; skip TF and keep as-is.
            if self.use_pixel_frame:
                pose_in_map = (float(pose_in_camera.x), float(pose_in_camera.y), float(pose_in_camera.z))
            else:
                # Get the transform from camera frame to fixed frame
                transform = self.tf_buffer.lookup_transform(
                    fixed_frame,
                    camera_frame,
                    Time(sec=0, nanosec=0)
                )
                # Transform the pose to the fixed frame
                pose_in_map = self.transform_point(pose_in_camera, transform)

            # Prepare embeddings with empty handling: treat [] as None
            def _normalize_vec(vec):
                if vec is None:
                    return None
                arr = np.array(vec, dtype=np.float32).ravel()
                return arr if arr.size > 0 else None

            img_vec = _normalize_vec(embeddings)
            txt_vec = _normalize_vec(goal_embedding)

            # Prefer computing similarity with available vectors.
            # If both new image and text embeddings are present, use them.
            # Otherwise, compute with existing image embedding (on merge) handled below.
            similarity = float(np.dot(img_vec, txt_vec)) if (img_vec is not None and txt_vec is not None) else None

            # Iterate over stored objects and check if this detection is close to any.
            for existing_id, entry in self.objects.items():
                dist = self.euclidean_distance(pose_in_map, entry.pose_map)
                if dist < distance_threshold:
                    avg_pose = tuple(
                        (entry.pose_map[i] * entry.occurrences + pose_in_map[i]) / (entry.occurrences + 1)
                        for i in range(3)
                    )
                    # Preserve older embedding if new detection carries empty/None embeddings
                    new_image_embedding = img_vec if img_vec is not None else entry.image_embedding

                    # Update similarity: prefer new img+txt; else if only txt provided, use it with existing image embedding
                    if similarity is not None:
                        new_similarity = similarity
                    elif txt_vec is not None and entry.image_embedding is not None:
                        new_similarity = float(np.dot(entry.image_embedding, txt_vec))
                    else:
                        new_similarity = entry.similarity

                    self.objects[existing_id] = entry._replace(
                        pose_map=avg_pose,
                        occurrences=entry.occurrences+1,
                        similarity=new_similarity,
                        image_embedding=new_image_embedding
                    )
                    return False

            self.objects[object_id] = ObjectEntry(
                frame = camera_frame,
                timestamp = detection_stamp,
                pose_cam = (float(pose_in_camera.x), float(pose_in_camera.y), float(pose_in_camera.z)),
                pose_map = pose_in_map,
                occurrences = 1,
                name = object_name,
                similarity = similarity,
                image_embedding = img_vec
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
    
    def update_all_to_latest_map(self, fixed_frame: str = 'map') -> None:
        """
        Recompute pose_map for every stored object using the transform available in tf.

        This method attempts to re-transform each object's stored camera-frame pose
        into the fixed_frame using the transform at the object's stored timestamp.
        """
        if self.use_pixel_frame:
            # In RGB mode, pose_map is already in pixel coordinates; nothing to update.
            return
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
                    image_embedding=image_embedding
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
                'image_embedding': entry.image_embedding.tolist() if entry.image_embedding is not None else None
            }

        with open(path, 'w') as json_file:
            json.dump(export_data, json_file, indent=4)
        return