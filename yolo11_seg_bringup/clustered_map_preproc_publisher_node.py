import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from yolo11_seg_interfaces.msg import (
    ClusterBoundingBox2D,
    ClusterDimensions2D,
    ClusteredMapObject,
    ClusteredMapObjectArray,
)


class ClusteredMapPreprocPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("clustered_map_preproc_publisher_node")

        self.declare_parameter(
            "input_map_file",
            "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/map_v6.json",
        )
        self.declare_parameter(
            "output_clustered_map_file",
            "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/clustered_map_v6.json",
        )
        self.declare_parameter("output_topic", "/vision/clustered_map_v6")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("eps", 2.4)
        self.declare_parameter("min_samples", 2)
        self.declare_parameter("publish_rate_hz", 0.5)

        self.input_map_file = str(self.get_parameter("input_map_file").value)
        self.output_clustered_map_file = str(self.get_parameter("output_clustered_map_file").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.eps = float(self.get_parameter("eps").value)
        self.min_samples = int(self.get_parameter("min_samples").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        if self.publish_rate_hz <= 0.0:
            self.get_logger().warning("[DEBUG] publish_rate_hz <= 0.0, forcing to 0.5 Hz")
            self.publish_rate_hz = 0.5

        self.publisher = self.create_publisher(ClusteredMapObjectArray, self.output_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._process_publish_cycle)

        self.get_logger().info(
            "[DEBUG] ClusteredMapPreprocPublisherNode started: "
            f"input={self.input_map_file}, output={self.output_clustered_map_file}, "
            f"topic={self.output_topic}, eps={self.eps}, min_samples={self.min_samples}, "
            f"rate={self.publish_rate_hz} Hz"
        )

    def _load_house_map(self, file_path: str) -> Dict[str, Dict[str, object]]:
        self.get_logger().info(f"[DEBUG] Loading map from: {file_path}")
        path = Path(file_path)
        if not path.exists():
            self.get_logger().error(f"[DEBUG] Input map file does not exist: {file_path}")
            return {}

        try:
            with path.open("r", encoding="utf-8") as handle:
                house_map = json.load(handle)
        except Exception as exc:
            self.get_logger().error(f"[DEBUG] Failed loading map json: {exc}")
            return {}

        if not isinstance(house_map, dict):
            self.get_logger().error("[DEBUG] Input map JSON root must be an object/dict")
            return {}

        map_objects = {}
        for obj_id, obj in house_map.items():
            try:
                name = str(obj["name"])
                pose_map = obj["pose_map"]
                x = float(pose_map["x"])
                y = float(pose_map["y"])
                z = float(pose_map["z"])
                similarity = float(
                    obj.get(
                        "similarity",
                        obj.get("similarity_score", obj.get("clip_similarity", 0.0)),
                    )
                )
            except Exception as exc:
                self.get_logger().warning(
                    f"[DEBUG] Skipping object {obj_id}: invalid fields ({exc})"
                )
                continue

            map_objects[str(obj_id)] = {
                "name": name,
                "coords": (x, y, z),
                "similarity": similarity,
            }

        self.get_logger().info(f"[DEBUG] Loaded {len(map_objects)} objects from map")
        return map_objects

    def _cluster_map(self, object_dict: Dict[str, Dict[str, object]]) -> List[dict]:
        self.get_logger().info(
            f"[DEBUG] Starting clustering with eps={self.eps}, min_samples={self.min_samples}"
        )
        if not object_dict:
            self.get_logger().warning("[DEBUG] Empty object dictionary")
            return []

        obj_ids = list(object_dict.keys())
        coords = [obj["coords"] for obj in object_dict.values()]
        names = [obj["name"] for obj in object_dict.values()]
        similarities = [float(obj.get("similarity", 0.0)) for obj in object_dict.values()]
        X = np.array(coords)
        self.get_logger().info(f"[DEBUG] Prepared {len(obj_ids)} objects for clustering")

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean").fit(X)
        labels = clustering.labels_
        self.get_logger().info(f"[DEBUG] DBSCAN completed. Labels={labels.tolist()}")

        outlier_indices = np.where(labels == -1)[0]
        if len(outlier_indices) > 0:
            max_cluster_id = max(labels[labels != -1]) if len(labels[labels != -1]) > 0 else -1
            next_cluster_id = int(max_cluster_id) + 1
            for i, outlier_idx in enumerate(outlier_indices):
                labels[outlier_idx] = next_cluster_id + i
                self.get_logger().info(
                    "[DEBUG] Reassigned outlier "
                    f"index={int(outlier_idx)} to cluster={int(next_cluster_id + i)}"
                )

        unique_labels = sorted(set(labels.tolist()))
        centroid_by_label = {}
        dimensions_by_label = {}

        for lbl in unique_labels:
            indices = np.where(labels == lbl)[0]
            cluster_points = X[indices]
            centroid = cluster_points.mean(axis=0)

            centroid_by_label[int(lbl)] = {
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]),
            }

            min_coords = cluster_points.min(axis=0)
            max_coords = cluster_points.max(axis=0)
            width = float(max_coords[0] - min_coords[0])
            length = float(max_coords[1] - min_coords[1])

            cluster_points_2d = cluster_points[:, :2]
            centroid_2d = centroid[:2]
            distances = np.linalg.norm(cluster_points_2d - centroid_2d, axis=1)
            radius = float(np.max(distances))

            dimensions_by_label[int(lbl)] = {
                "bounding_box": {
                    "min": {"x": float(min_coords[0]), "y": float(min_coords[1])},
                    "max": {"x": float(max_coords[0]), "y": float(max_coords[1])},
                    "dimensions": {"width": width, "length": length},
                },
                "radius": radius,
            }

            self.get_logger().info(
                "[DEBUG] Cluster "
                f"{int(lbl)} centroid=({centroid_by_label[int(lbl)]['x']:.2f}, "
                f"{centroid_by_label[int(lbl)]['y']:.2f}, {centroid_by_label[int(lbl)]['z']:.2f}) "
                f"bbox={width:.2f}x{length:.2f}m radius={radius:.2f}m"
            )

        final_output = []
        for obj_id, name, coord, label, similarity in zip(
            obj_ids, names, coords, labels, similarities
        ):
            final_output.append(
                {
                    "id": obj_id,
                    "cluster": int(label),
                    "class": name,
                    "similarity": float(similarity),
                    "coords": {
                        "x": float(coord[0]),
                        "y": float(coord[1]),
                        "z": float(coord[2]),
                    },
                    "cluster_centroid": centroid_by_label[int(label)],
                    "cluster_dimensions": dimensions_by_label[int(label)],
                }
            )

            self.get_logger().info(
                f"[DEBUG] Object '{obj_id}' (class={name}) assigned to cluster={int(label)}"
            )

        self.get_logger().info(
            f"[DEBUG] Clustering complete. Final output has {len(final_output)} objects"
        )
        return final_output

    def _write_map_to_file(self, clustered_map: List[dict], file_path: str) -> None:
        self.get_logger().info(f"[DEBUG] Writing {len(clustered_map)} entries to: {file_path}")
        path = Path(file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(clustered_map, handle, indent=4)
            self.get_logger().info("[DEBUG] Successfully wrote clustered map to file")
        except Exception as exc:
            self.get_logger().error(f"[DEBUG] Failed writing clustered map: {exc}")

    @staticmethod
    def _vec3(x: float, y: float, z: float) -> Vector3:
        msg = Vector3()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        return msg

    def _to_msg(self, clustered_map: List[dict]) -> ClusteredMapObjectArray:
        out_msg = ClusteredMapObjectArray()
        out_msg.stamp = self.get_clock().now().to_msg()
        out_msg.frame_id = self.frame_id

        for entry in clustered_map:
            obj_msg = ClusteredMapObject()
            obj_msg.id = str(entry.get("id", ""))
            obj_msg.cluster = int(entry.get("cluster", -1))
            obj_msg.class_name = str(entry.get("class", "unknown"))
            obj_msg.similarity = float(entry.get("similarity", 0.0))

            coords = entry.get("coords", {})
            obj_msg.coords = self._vec3(
                float(coords.get("x", 0.0)),
                float(coords.get("y", 0.0)),
                float(coords.get("z", 0.0)),
            )

            centroid = entry.get("cluster_centroid", {})
            obj_msg.cluster_centroid = self._vec3(
                float(centroid.get("x", 0.0)),
                float(centroid.get("y", 0.0)),
                float(centroid.get("z", 0.0)),
            )

            dims = entry.get("cluster_dimensions", {})
            bbox = dims.get("bounding_box", {})
            bbox_min = bbox.get("min", {})
            bbox_max = bbox.get("max", {})
            bbox_dims = bbox.get("dimensions", {})

            bbox_msg = ClusterBoundingBox2D()
            bbox_msg.min_x = float(bbox_min.get("x", 0.0))
            bbox_msg.min_y = float(bbox_min.get("y", 0.0))
            bbox_msg.max_x = float(bbox_max.get("x", 0.0))
            bbox_msg.max_y = float(bbox_max.get("y", 0.0))
            bbox_msg.width = float(bbox_dims.get("width", 0.0))
            bbox_msg.length = float(bbox_dims.get("length", 0.0))

            dims_msg = ClusterDimensions2D()
            dims_msg.bounding_box = bbox_msg
            dims_msg.radius = float(dims.get("radius", 0.0))

            obj_msg.cluster_dimensions = dims_msg
            out_msg.objects.append(obj_msg)

        return out_msg

    def _process_publish_cycle(self) -> None:
        clean_map = self._load_house_map(self.input_map_file)
        clustered_map = self._cluster_map(clean_map)

        self._write_map_to_file(clustered_map, self.output_clustered_map_file)

        msg = self._to_msg(clustered_map)
        self.publisher.publish(msg)
        self.get_logger().info(
            "[DEBUG] Published clustered map message "
            f"with {len(msg.objects)} objects on {self.output_topic}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusteredMapPreprocPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
