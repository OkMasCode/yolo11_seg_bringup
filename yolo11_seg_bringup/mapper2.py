from geometry_msgs.msg import TransformStamped, Vector3, PointStamped
from tf2_ros import Buffer, TransformException
import tf2_geometry_msgs
import rclpy
from rclpy.node import Node
from collections import namedtuple
import math
import csv
import numpy as np
from builtin_interfaces.msg import Time
import os
from typing import Dict, Tuple, List, Any

ObjectEntry = namedtuple('ObjectEntry', ['frame', 'timestamp', 'pose_cam', 'pose_map', 'occurrences', 'name'])

_EPS = np.finfo(float).eps * 4.0 # Small epsilon value to avoid division by zero in quaternion normalization

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

class SemanticObjectMap:
    """
    A class to maintain a semantic map of detected objects, merging detections that are close to each other.
    Next update is to add image encodings of the detected object
    """
    def __init__(self, tf_buffer: Buffer, node: Node):
        self.objects = {}
        self.tf_buffer = tf_buffer
        self.node = node

    def add_detection(self, object_name: str, object_id: str, pose_in_camera, detection_stamp, camera_frame='camera3_color_optical_frame', fixed_frame='map', distance_threshold=0.2):
        try:
            # Request the transform that maps points from camera_frame into fixed_frame.
            # Passing Time(sec=0, nanosec=0) is intended to request the latest transform.
            # NOTE: using the detection timestamp may instead be desired. This code uses
            # the tf buffer's "latest" transform.
            transform = self.tf_buffer.lookup_transform(
                fixed_frame,
                camera_frame,
                Time(sec=0, nanosec=0) # rclpy.time.Time.from_msg(detection_stamp) 
            ) # retrieve the transform from fixed frame to camera frame in the instant the detection happened

            # Convert the point expressed in camera_frame into coordinates in fixed_frame.
            new_pose_map = self.transform_point(pose_in_camera, transform) # get the coordinates of the object in the absolute frame

            # Iterate over stored objects and check if this detection is close to any.
            # If it is within distance_threshold we merge it (average positions).
            for existing_id, entry in self.objects.items():
                dist = self.euclidean_distance(new_pose_map, entry.pose_map)
                if dist < distance_threshold:
                    # Compute a simple average of the existing stored pose and the new pose.
                    avg_pose = tuple(
                        (entry.pose_map[i] * entry.occurrences + new_pose_map[i]) / (entry.occurrences + 1)
                        for i in range(3)
                    )
                    # Replace the stored entry winewth a  entry that increments occurrences
                    self.objects[existing_id] = entry._replace(pose_map=avg_pose, occurrences=entry.occurrences+1)
                    # Log a message that we merged detections
                    self.node.get_logger().info(f"Merged {object_id} into {existing_id}")
                    return False

            # If we didn't merge, create a new ObjectEntry and store it
            self.objects[object_id] = ObjectEntry(
                frame=camera_frame,
                timestamp=detection_stamp,
                # store the raw camera coordinates as a tuple for later re-projection
                pose_cam=(pose_in_camera.x, pose_in_camera.y, pose_in_camera.z),
                # store the computed map-frame pose
                pose_map=new_pose_map,
                occurrences=1,
                name=object_name
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
        # Extract quaternion and translation from the TransformStamped message. The
        # quaternion encodes rotation of the camera frame w.r.t. the absolute frame.
        q = transform.transform.rotation # rotation of the camera frame with respect to the absolute frame
        t = transform.transform.translation # translation of the camera frame with respect to the absolute frame

        # Convert quaternion to a 4x4 matrix then take the upper-left 3x3 rotation matrix
        # The quaternion_matrix function expects 4 values; this code provides [x,y,z,w].
        rot_matrix = quaternion_matrix([q.w, q.x, q.y, q.z])[:3, :3] # obtains the rotation matrix from the quaternion values

        # Convert the Vector3 point into a 3x1 numpy column vector
        point_vec = np.array([[point.x], [point.y], [point.z]]) # this is the coordinates of the object in camera frame
        # Convert the translation into a 3x1 numpy column vector
        translation = np.array([[t.x], [t.y], [t.z]]) # this is the coordinates of the camera in abosulte frame

        # Apply rotation then add translation: result = R * point + t
        result = rot_matrix @ point_vec + translation 
        # Return as a simple (x, y, z) tuple
        return (result[0, 0], result[1, 0], result[2, 0])

    def get_map(self) -> List[str]:
        """Return a list of object IDs currently stored in the map."""
        return [id_entry for id_entry, _ in list(self.objects.items())]

    def update_all_to_latest_map(self, fixed_frame: str = 'map') -> None:
        """
        Recompute pose_map for every stored object using the transform available in tf.

        This method attempts to re-transform each object's stored camera-frame pose
        into the fixed_frame using the transform at the object's stored timestamp.
        Depending on desired semantics, you may want to request the latest transform
        instead of the transform at the detection time.
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
                # Update the stored entry's pose_map with the recomputed pose
                self.objects[object_id] = entry._replace(pose_map=new_pose)

            except TransformException as ex:
                self.node.get_logger().warning(f"Could not update {object_id}: {ex}")

    def euclidean_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Compute Euclidean distance between two 3D points represented as (x,y,z)."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    
    def export_to_csv(self, directory_path, file='detections.csv'):
        """
        Export the current object map to a CSV file.

        Format: class, x, y, z

        This method first calls update_all_to_latest_map() to reproject stored camera poses
        into the fixed frame (using transforms at the stored timestamps).
        """
        # Refresh stored poses before exporting
        self.update_all_to_latest_map()
        os.makedirs(directory_path, exist_ok=True)
        path = os.path.join(directory_path, file)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for _, entry in list(self.objects.items()):
                x, y, z = entry.pose_map
                writer.writerow([entry.name, x, y, z])
        return
