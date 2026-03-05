import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Vector3
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException

_EPS = np.finfo(float).eps * 4.0


@dataclass
class MapObject:
    """Long-lived object stored in the semantic map."""

    map_id: str
    frame: str
    timestamp: Time
    pose_cam: Tuple[float, float, float]
    pose_map: Tuple[float, float, float]
    box_size: Tuple[float, float, float]
    occurrences: int
    first_seen_ns: int
    last_seen_ns: int
    current_name: str
    class_votes: Dict[str, float] = field(default_factory=dict)
    similarity: float = 0.0
    confidence_ema: float = 0.0
    image_embedding: Optional[np.ndarray] = None
    source_track_id: Optional[str] = None


@dataclass
class TentativeTrack:
    """Short-lived candidate that needs repeated evidence before promotion."""

    track_id: str
    frame: str
    timestamp: Time
    pose_cam: Tuple[float, float, float]
    pose_map: Tuple[float, float, float]
    box_size: Tuple[float, float, float]
    hits: int
    first_seen_ns: int
    last_seen_ns: int
    class_name: str
    class_votes: Dict[str, float] = field(default_factory=dict)
    confidence_max: float = 0.0
    similarity: float = 0.0
    image_embedding: Optional[np.ndarray] = None


@dataclass
class TrackBinding:
    """Association between detector tracker IDs and persistent map objects."""

    track_id: str
    map_id: str
    first_seen_ns: int
    last_seen_ns: int
    stable_hits: int = 1
    misses: int = 0


# ----------------- Geometry helpers ------------------ #


def quaternion_matrix(quaternion):
    """Convert [w, x, y, z] quaternion to a 4x4 homogeneous matrix."""
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


# -------------------- Mapper ------------------- #


class SemanticObjectMapV2:
    """
    Persistent semantic mapper that leverages upstream tracker IDs.

    Design goals:
    1) Track ID is treated as short-term identity from the vision node.
    2) Mapper keeps long-term map IDs that survive ID switches and class jitter.
    3) Associations are validated with adaptive geometric gates.
    """

    def __init__(self, tf_buffer: Buffer, node: Node):
        self.tf_buffer = tf_buffer
        self.node = node

        # Persistent map objects keyed by map_id.
        self.objects: Dict[str, MapObject] = {}
        # Short-lived objects waiting for confirmation.
        self.tentative_tracks: Dict[str, TentativeTrack] = {}
        # Active mapping between detector track_id and map_id.
        self.track_bindings: Dict[str, TrackBinding] = {}

        self._next_map_id = 1

        # Confirmation/lifecycle policy.
        self.confirmation_min_hits = 3
        self.confirmation_time_window_sec = 1.5
        self.tentative_max_stale_sec = 1.5
        self.binding_ttl_sec = 2.5

        # Adaptive gate parameters. The effective gate grows with object size.
        self.base_distance_gate = 0.20
        self.size_distance_gain = 0.30
        self.max_distance_gate = 1.20

        # When class mismatch happens, we allow update only if geometry is very strong.
        self.class_mismatch_penalty = 0.25

        # Periodic dedup/fusion among confirmed map objects.
        self.merge_min_overlap = 0.30
        self.merge_max_distance = 0.35

        # EMA coefficient for pose and confidence updates.
        self.pose_ema_alpha = 0.25
        self.confidence_ema_alpha = 0.20

        # Runtime statistics to compare performance against the baseline mapper.
        self.stats = {
            'total_detections': 0,
            'updated_via_binding': 0,
            'updated_via_rebind': 0,
            'created_new_object': 0,
            'tentative_updates': 0,
            'dedup_merges': 0,
            'binding_invalidations': 0,
        }

    def add_detection(
        self,
        object_name: str,
        tracker_id: str,
        pose_in_camera,
        detection_stamp,
        camera_frame: str = 'camera_color_optical_frame',
        fixed_frame: str = 'map',
        embeddings=None,
        similarity: float = 0.0,
        confidence: float = 0.0,
        box_min=None,
        box_max=None,
    ) -> bool:
        """
        Ingest one detection and update the long-term map.

        Returns True if a new persistent map object was created.
        """
        self.stats['total_detections'] += 1

        try:
            lookup_time = rclpy.time.Time.from_msg(detection_stamp)
            transform = self.tf_buffer.lookup_transform(
                fixed_frame,
                camera_frame,
                lookup_time,
                timeout=Duration(seconds=0.8),
            )
            pose_in_map = self.transform_point(pose_in_camera, transform)
        except TransformException as ex:
            self.node.get_logger().warning(f"[mapper_v2] Could not transform detection: {ex}")
            return False

        current_ns = self._stamp_to_ns(detection_stamp)
        self._prune_stale_state(current_ns)

        new_size = self._compute_box_size(box_min, box_max)
        img_vec = self._to_embedding(embeddings)

        # 1) Fast path: update through existing track->map binding.
        binding = self.track_bindings.get(tracker_id)
        if binding and binding.map_id in self.objects:
            obj = self.objects[binding.map_id]
            if self._is_valid_association(obj, pose_in_map, new_size, object_name):
                self._update_object(
                    map_id=obj.map_id,
                    object_name=object_name,
                    detection_stamp=detection_stamp,
                    pose_in_camera=(pose_in_camera.x, pose_in_camera.y, pose_in_camera.z),
                    pose_in_map=pose_in_map,
                    box_size=new_size,
                    similarity=similarity,
                    confidence=confidence,
                    image_embedding=img_vec,
                    current_ns=current_ns,
                    source_track_id=tracker_id,
                )
                binding.last_seen_ns = current_ns
                binding.stable_hits += 1
                binding.misses = 0
                self.stats['updated_via_binding'] += 1
                return False

            # Binding exists but geometry says it is no longer valid.
            binding.misses += 1
            self.stats['binding_invalidations'] += 1
            if binding.misses >= 2:
                del self.track_bindings[tracker_id]

        # 2) Rebind path: try to map this tracker to an existing map object.
        candidate_id = self._find_best_map_candidate(pose_in_map, new_size, object_name)
        if candidate_id is not None:
            self.track_bindings[tracker_id] = TrackBinding(
                track_id=tracker_id,
                map_id=candidate_id,
                first_seen_ns=current_ns,
                last_seen_ns=current_ns,
            )
            self._update_object(
                map_id=candidate_id,
                object_name=object_name,
                detection_stamp=detection_stamp,
                pose_in_camera=(pose_in_camera.x, pose_in_camera.y, pose_in_camera.z),
                pose_in_map=pose_in_map,
                box_size=new_size,
                similarity=similarity,
                confidence=confidence,
                image_embedding=img_vec,
                current_ns=current_ns,
                source_track_id=tracker_id,
            )
            self.stats['updated_via_rebind'] += 1
            return False

        # 3) No binding and no candidate: update tentative evidence for this track.
        created = self._update_tentative(
            object_name=object_name,
            tracker_id=tracker_id,
            pose_in_camera=pose_in_camera,
            pose_in_map=pose_in_map,
            box_size=new_size,
            detection_stamp=detection_stamp,
            similarity=similarity,
            confidence=confidence,
            image_embedding=img_vec,
            current_ns=current_ns,
            frame=camera_frame,
        )

        # Run lightweight dedup only when the map changes.
        if created:
            self._merge_duplicates()

        return created

    # ----------------- Core update methods ------------------ #

    def _update_tentative(
        self,
        object_name: str,
        tracker_id: str,
        pose_in_camera,
        pose_in_map,
        box_size,
        detection_stamp,
        similarity: float,
        confidence: float,
        image_embedding,
        current_ns: int,
        frame: str,
    ) -> bool:
        """Aggregate tentative evidence and promote to persistent map object."""
        self.stats['tentative_updates'] += 1
        tentative = self.tentative_tracks.get(tracker_id)
        pose_cam = (pose_in_camera.x, pose_in_camera.y, pose_in_camera.z)

        if tentative is None:
            self.tentative_tracks[tracker_id] = TentativeTrack(
                track_id=tracker_id,
                frame=frame,
                timestamp=detection_stamp,
                pose_cam=pose_cam,
                pose_map=pose_in_map,
                box_size=box_size,
                hits=1,
                first_seen_ns=current_ns,
                last_seen_ns=current_ns,
                class_name=object_name,
                class_votes={object_name: max(confidence, 0.01)},
                confidence_max=confidence,
                similarity=similarity,
                image_embedding=image_embedding,
            )
            return False

        # If the tracker jumped too far, reset tentative evidence.
        gate = self._dynamic_gate(tentative.box_size, box_size)
        dist = self.euclidean_distance(pose_in_map, tentative.pose_map)
        overlap = self.check_aabb_intersection(
            center_a=pose_in_map,
            size_a=box_size,
            center_b=tentative.pose_map,
            size_b=tentative.box_size,
        )
        if dist > gate and not overlap:
            self.tentative_tracks[tracker_id] = TentativeTrack(
                track_id=tracker_id,
                frame=frame,
                timestamp=detection_stamp,
                pose_cam=pose_cam,
                pose_map=pose_in_map,
                box_size=box_size,
                hits=1,
                first_seen_ns=current_ns,
                last_seen_ns=current_ns,
                class_name=object_name,
                class_votes={object_name: max(confidence, 0.01)},
                confidence_max=confidence,
                similarity=similarity,
                image_embedding=image_embedding,
            )
            return False

        # Merge observation into tentative track.
        hits = tentative.hits + 1
        merged_pose = self._ema_tuple(tentative.pose_map, pose_in_map, self.pose_ema_alpha)
        merged_cam = self._ema_tuple(tentative.pose_cam, pose_cam, self.pose_ema_alpha)
        merged_size = self._ema_tuple(tentative.box_size, box_size, self.pose_ema_alpha)
        votes = dict(tentative.class_votes)
        votes[object_name] = votes.get(object_name, 0.0) + max(confidence, 0.01)

        self.tentative_tracks[tracker_id] = TentativeTrack(
            track_id=tracker_id,
            frame=frame,
            timestamp=detection_stamp,
            pose_cam=merged_cam,
            pose_map=merged_pose,
            box_size=merged_size,
            hits=hits,
            first_seen_ns=tentative.first_seen_ns,
            last_seen_ns=current_ns,
            class_name=max(votes.items(), key=lambda x: x[1])[0],
            class_votes=votes,
            confidence_max=max(tentative.confidence_max, confidence),
            similarity=max(tentative.similarity, similarity),
            image_embedding=(image_embedding if image_embedding is not None else tentative.image_embedding),
        )

        window_ns = int(self.confirmation_time_window_sec * 1e9)
        if hits >= self.confirmation_min_hits and (current_ns - tentative.first_seen_ns) <= window_ns:
            promoted = self.tentative_tracks.pop(tracker_id)
            map_id = self._new_map_id()
            self.objects[map_id] = MapObject(
                map_id=map_id,
                frame=promoted.frame,
                timestamp=promoted.timestamp,
                pose_cam=promoted.pose_cam,
                pose_map=promoted.pose_map,
                box_size=promoted.box_size,
                occurrences=promoted.hits,
                first_seen_ns=promoted.first_seen_ns,
                last_seen_ns=current_ns,
                current_name=promoted.class_name,
                class_votes=promoted.class_votes,
                similarity=promoted.similarity,
                confidence_ema=promoted.confidence_max,
                image_embedding=promoted.image_embedding,
                source_track_id=tracker_id,
            )
            self.track_bindings[tracker_id] = TrackBinding(
                track_id=tracker_id,
                map_id=map_id,
                first_seen_ns=current_ns,
                last_seen_ns=current_ns,
            )
            self.stats['created_new_object'] += 1
            return True

        return False

    def _update_object(
        self,
        map_id: str,
        object_name: str,
        detection_stamp,
        pose_in_camera,
        pose_in_map,
        box_size,
        similarity: float,
        confidence: float,
        image_embedding,
        current_ns: int,
        source_track_id: str,
    ) -> None:
        """Update a persistent object with one new observation using smoothed estimates."""
        entry = self.objects[map_id]

        merged_pose = self._ema_tuple(entry.pose_map, pose_in_map, self.pose_ema_alpha)
        merged_cam = self._ema_tuple(entry.pose_cam, pose_in_camera, self.pose_ema_alpha)
        merged_size = self._ema_tuple(entry.box_size, box_size, self.pose_ema_alpha)

        class_votes = dict(entry.class_votes)
        class_votes[object_name] = class_votes.get(object_name, 0.0) + max(confidence, 0.01)
        best_class = max(class_votes.items(), key=lambda x: x[1])[0]

        conf_ema = (1.0 - self.confidence_ema_alpha) * entry.confidence_ema + self.confidence_ema_alpha * confidence

        if image_embedding is not None and confidence >= entry.confidence_ema:
            embedding = image_embedding
        else:
            embedding = entry.image_embedding

        self.objects[map_id] = MapObject(
            map_id=entry.map_id,
            frame=entry.frame,
            timestamp=detection_stamp,
            pose_cam=merged_cam,
            pose_map=merged_pose,
            box_size=merged_size,
            occurrences=entry.occurrences + 1,
            first_seen_ns=entry.first_seen_ns,
            last_seen_ns=current_ns,
            current_name=best_class,
            class_votes=class_votes,
            similarity=max(entry.similarity, similarity),
            confidence_ema=conf_ema,
            image_embedding=embedding,
            source_track_id=source_track_id,
        )

    # ----------------- Association helpers ------------------ #

    def _is_valid_association(self, entry: MapObject, pose_map, box_size, class_name: str) -> bool:
        """Validate whether detection is compatible with an existing bound map object."""
        gate = self._dynamic_gate(entry.box_size, box_size)
        dist = self.euclidean_distance(pose_map, entry.pose_map)
        overlap = self.check_aabb_intersection(
            center_a=pose_map,
            size_a=box_size,
            center_b=entry.pose_map,
            size_b=entry.box_size,
        )

        # If class differs, require stronger geometry evidence.
        if class_name != entry.current_name:
            return dist <= (gate * (1.0 - self.class_mismatch_penalty)) and overlap

        return dist <= gate or overlap

    def _find_best_map_candidate(self, pose_map, box_size, class_name: str) -> Optional[str]:
        """Select best existing map object for re-binding an unbound tracker."""
        best_id = None
        best_score = float('inf')

        for map_id, entry in self.objects.items():
            gate = self._dynamic_gate(entry.box_size, box_size)
            dist = self.euclidean_distance(pose_map, entry.pose_map)
            if dist > gate:
                continue

            size_cost = self._size_cost(entry.box_size, box_size)
            class_cost = 0.0 if class_name == entry.current_name else 0.5
            score = 0.6 * (dist / max(gate, 1e-6)) + 0.25 * size_cost + 0.15 * class_cost

            if score < best_score:
                best_score = score
                best_id = map_id

        # Conservative score threshold avoids accidental re-binds.
        if best_id is not None and best_score <= 0.85:
            return best_id
        return None

    def _merge_duplicates(self) -> None:
        """Fuse confirmed objects that are almost certainly duplicates."""
        map_ids = list(self.objects.keys())
        merged_pairs = []

        for i in range(len(map_ids)):
            a_id = map_ids[i]
            if a_id not in self.objects:
                continue
            for j in range(i + 1, len(map_ids)):
                b_id = map_ids[j]
                if b_id not in self.objects:
                    continue

                a = self.objects[a_id]
                b = self.objects[b_id]
                dist = self.euclidean_distance(a.pose_map, b.pose_map)
                overlap = self.check_aabb_intersection(a.pose_map, a.box_size, b.pose_map, b.box_size)
                if dist > self.merge_max_distance or not overlap:
                    continue

                overlap_ratio = self._aabb_overlap_ratio(a.pose_map, a.box_size, b.pose_map, b.box_size)
                if overlap_ratio < self.merge_min_overlap:
                    continue

                # Keep the older map object as canonical and absorb the newer one.
                keep_id, drop_id = (a_id, b_id) if a.first_seen_ns <= b.first_seen_ns else (b_id, a_id)
                self._fuse_objects(keep_id, drop_id)
                merged_pairs.append((keep_id, drop_id))

        if merged_pairs:
            self.stats['dedup_merges'] += len(merged_pairs)
            self.node.get_logger().info(f"[mapper_v2] merged duplicate pairs: {merged_pairs}")

    def _fuse_objects(self, keep_id: str, drop_id: str) -> None:
        """Merge two map objects into one and redirect any tracker bindings."""
        keep = self.objects[keep_id]
        drop = self.objects[drop_id]

        total_hits = keep.occurrences + drop.occurrences
        w_keep = keep.occurrences / max(total_hits, 1)
        w_drop = drop.occurrences / max(total_hits, 1)

        fused_pose = (
            keep.pose_map[0] * w_keep + drop.pose_map[0] * w_drop,
            keep.pose_map[1] * w_keep + drop.pose_map[1] * w_drop,
            keep.pose_map[2] * w_keep + drop.pose_map[2] * w_drop,
        )
        fused_cam = (
            keep.pose_cam[0] * w_keep + drop.pose_cam[0] * w_drop,
            keep.pose_cam[1] * w_keep + drop.pose_cam[1] * w_drop,
            keep.pose_cam[2] * w_keep + drop.pose_cam[2] * w_drop,
        )
        fused_size = (
            keep.box_size[0] * w_keep + drop.box_size[0] * w_drop,
            keep.box_size[1] * w_keep + drop.box_size[1] * w_drop,
            keep.box_size[2] * w_keep + drop.box_size[2] * w_drop,
        )

        votes = dict(keep.class_votes)
        for name, value in drop.class_votes.items():
            votes[name] = votes.get(name, 0.0) + value
        best_class = max(votes.items(), key=lambda x: x[1])[0]

        self.objects[keep_id] = MapObject(
            map_id=keep.map_id,
            frame=keep.frame,
            timestamp=keep.timestamp if keep.last_seen_ns >= drop.last_seen_ns else drop.timestamp,
            pose_cam=fused_cam,
            pose_map=fused_pose,
            box_size=fused_size,
            occurrences=total_hits,
            first_seen_ns=min(keep.first_seen_ns, drop.first_seen_ns),
            last_seen_ns=max(keep.last_seen_ns, drop.last_seen_ns),
            current_name=best_class,
            class_votes=votes,
            similarity=max(keep.similarity, drop.similarity),
            confidence_ema=max(keep.confidence_ema, drop.confidence_ema),
            image_embedding=keep.image_embedding if keep.image_embedding is not None else drop.image_embedding,
            source_track_id=keep.source_track_id,
        )

        del self.objects[drop_id]

        # Redirect any active track bindings that pointed to dropped object.
        for track_id, binding in list(self.track_bindings.items()):
            if binding.map_id == drop_id:
                self.track_bindings[track_id] = TrackBinding(
                    track_id=track_id,
                    map_id=keep_id,
                    first_seen_ns=binding.first_seen_ns,
                    last_seen_ns=binding.last_seen_ns,
                    stable_hits=binding.stable_hits,
                    misses=0,
                )

    # ----------------- State management ------------------ #

    def _prune_stale_state(self, current_ns: int) -> None:
        """Drop stale tentative tracks and old bindings to keep state healthy."""
        stale_tentative_ns = int(self.tentative_max_stale_sec * 1e9)
        stale_binding_ns = int(self.binding_ttl_sec * 1e9)

        for track_id, tentative in list(self.tentative_tracks.items()):
            if current_ns - tentative.last_seen_ns > stale_tentative_ns:
                del self.tentative_tracks[track_id]

        for track_id, binding in list(self.track_bindings.items()):
            if current_ns - binding.last_seen_ns > stale_binding_ns:
                del self.track_bindings[track_id]

    def _new_map_id(self) -> str:
        map_id = f"map_obj_{self._next_map_id:06d}"
        self._next_map_id += 1
        return map_id

    # ----------------- Serialization ------------------ #

    def export_to_json(self, directory_path: str, file: str = 'map_v2.json') -> None:
        """Export map objects and mapper stats for reproducible benchmarking."""
        os.makedirs(directory_path, exist_ok=True)
        path = os.path.join(directory_path, file)

        payload = {
            'schema_version': 2,
            'stats': self.stats,
            'objects': {},
        }

        for map_id, obj in self.objects.items():
            payload['objects'][map_id] = {
                'name': obj.current_name,
                'frame': obj.frame,
                'timestamp': {'sec': obj.timestamp.sec, 'nanosec': obj.timestamp.nanosec},
                'pose_map': {'x': float(obj.pose_map[0]), 'y': float(obj.pose_map[1]), 'z': float(obj.pose_map[2])},
                'pose_cam': {'x': float(obj.pose_cam[0]), 'y': float(obj.pose_cam[1]), 'z': float(obj.pose_cam[2])},
                'box_size': {'x': float(obj.box_size[0]), 'y': float(obj.box_size[1]), 'z': float(obj.box_size[2])},
                'occurrences': int(obj.occurrences),
                'first_seen_ns': int(obj.first_seen_ns),
                'last_seen_ns': int(obj.last_seen_ns),
                'similarity': float(obj.similarity),
                'confidence_ema': float(obj.confidence_ema),
                'class_votes': obj.class_votes,
                'source_track_id': obj.source_track_id,
                'image_embedding': obj.image_embedding.tolist() if obj.image_embedding is not None else None,
            }

        with open(path, 'w') as json_file:
            json.dump(payload, json_file, indent=4)

    def load_from_json(self, directory_path: str, file: str = 'map_v2.json') -> None:
        """Load exported v2 map data while preserving map object identities."""
        path = os.path.join(directory_path, file)
        if not os.path.exists(path):
            self.node.get_logger().info(f"[mapper_v2] No map file found at {path}, starting empty")
            return

        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)

            loaded_objects = data.get('objects', {})
            self.objects = {}

            max_numeric_id = 0
            for map_id, obj_data in loaded_objects.items():
                ts = Time(sec=int(obj_data['timestamp']['sec']), nanosec=int(obj_data['timestamp']['nanosec']))
                pose_map = (
                    float(obj_data['pose_map']['x']),
                    float(obj_data['pose_map']['y']),
                    float(obj_data['pose_map']['z']),
                )
                pose_cam = (
                    float(obj_data.get('pose_cam', {}).get('x', pose_map[0])),
                    float(obj_data.get('pose_cam', {}).get('y', pose_map[1])),
                    float(obj_data.get('pose_cam', {}).get('z', pose_map[2])),
                )
                box_data = obj_data.get('box_size', {'x': 0.1, 'y': 0.1, 'z': 0.1})
                box_size = (float(box_data['x']), float(box_data['y']), float(box_data['z']))

                img = obj_data.get('image_embedding')
                image_embedding = np.array(img, dtype=np.float32) if img else None

                self.objects[map_id] = MapObject(
                    map_id=map_id,
                    frame=obj_data['frame'],
                    timestamp=ts,
                    pose_cam=pose_cam,
                    pose_map=pose_map,
                    box_size=box_size,
                    occurrences=int(obj_data.get('occurrences', 1)),
                    first_seen_ns=int(obj_data.get('first_seen_ns', 0)),
                    last_seen_ns=int(obj_data.get('last_seen_ns', 0)),
                    current_name=obj_data.get('name', 'unknown'),
                    class_votes=obj_data.get('class_votes', {obj_data.get('name', 'unknown'): 1.0}),
                    similarity=float(obj_data.get('similarity', 0.0)),
                    confidence_ema=float(obj_data.get('confidence_ema', 0.0)),
                    image_embedding=image_embedding,
                    source_track_id=obj_data.get('source_track_id'),
                )

                if map_id.startswith('map_obj_'):
                    try:
                        max_numeric_id = max(max_numeric_id, int(map_id.split('_')[-1]))
                    except ValueError:
                        pass

            self._next_map_id = max_numeric_id + 1
            self.stats.update(data.get('stats', {}))
            self.node.get_logger().info(f"[mapper_v2] Loaded {len(self.objects)} objects from {path}")

        except Exception as ex:
            self.node.get_logger().error(f"[mapper_v2] Error loading map from {path}: {ex}")

    # ----------------- Utility methods ------------------ #

    def transform_point(self, point, transform: TransformStamped) -> Tuple[float, float, float]:
        """Apply rigid transform to a point from source frame into target frame."""
        q = transform.transform.rotation
        t = transform.transform.translation

        rot_matrix = quaternion_matrix([q.w, q.x, q.y, q.z])[:3, :3]
        point_vec = np.array([[point.x], [point.y], [point.z]])
        translation = np.array([[t.x], [t.y], [t.z]])
        result = rot_matrix @ point_vec + translation
        return (result[0, 0], result[1, 0], result[2, 0])

    def euclidean_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _dynamic_gate(self, size_a: Tuple[float, float, float], size_b: Tuple[float, float, float]) -> float:
        """Gate radius scales with larger object extent to handle large objects safely."""
        extent = max(max(size_a), max(size_b), 0.1)
        gate = self.base_distance_gate + self.size_distance_gain * extent
        return min(max(gate, self.base_distance_gate), self.max_distance_gate)

    def _size_cost(self, size_a: Tuple[float, float, float], size_b: Tuple[float, float, float]) -> float:
        """Normalized size discrepancy cost in [0, 1+] used in association score."""
        va = max(size_a[0] * size_a[1] * size_a[2], 1e-6)
        vb = max(size_b[0] * size_b[1] * size_b[2], 1e-6)
        ratio = max(va, vb) / min(va, vb)
        return min((ratio - 1.0), 2.0)

    def _compute_box_size(self, box_min, box_max) -> Tuple[float, float, float]:
        if box_min is None or box_max is None:
            return (0.10, 0.10, 0.10)
        return (
            max(abs(box_max[0] - box_min[0]), 0.01),
            max(abs(box_max[1] - box_min[1]), 0.01),
            max(abs(box_max[2] - box_min[2]), 0.01),
        )

    def _to_embedding(self, embeddings) -> Optional[np.ndarray]:
        if embeddings is None:
            return None
        arr = np.asarray(embeddings, dtype=np.float32)
        return arr if arr.size > 0 else None

    def _stamp_to_ns(self, stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _ema_tuple(self, old, new, alpha: float):
        return tuple((1.0 - alpha) * old[i] + alpha * new[i] for i in range(3))

    def check_aabb_intersection(self, center_a, size_a, center_b, size_b) -> bool:
        """Standard AABB intersection check used for geometric compatibility."""
        ha = np.array(size_a) / 2.0
        hb = np.array(size_b) / 2.0

        min_a = np.array(center_a) - ha
        max_a = np.array(center_a) + ha
        min_b = np.array(center_b) - hb
        max_b = np.array(center_b) + hb

        overlap_x = (min_a[0] <= max_b[0]) and (max_a[0] >= min_b[0])
        overlap_y = (min_a[1] <= max_b[1]) and (max_a[1] >= min_b[1])
        overlap_z = (min_a[2] <= max_b[2]) and (max_a[2] >= min_b[2])
        return overlap_x and overlap_y and overlap_z

    def _aabb_overlap_ratio(self, center_a, size_a, center_b, size_b) -> float:
        """Approximate overlap ratio for dedup decision (intersection / smaller volume)."""
        ha = np.array(size_a) / 2.0
        hb = np.array(size_b) / 2.0

        min_a = np.array(center_a) - ha
        max_a = np.array(center_a) + ha
        min_b = np.array(center_b) - hb
        max_b = np.array(center_b) + hb

        inter_min = np.maximum(min_a, min_b)
        inter_max = np.minimum(max_a, max_b)
        inter_size = np.maximum(inter_max - inter_min, 0.0)
        inter_vol = inter_size[0] * inter_size[1] * inter_size[2]

        vol_a = max(size_a[0] * size_a[1] * size_a[2], 1e-6)
        vol_b = max(size_b[0] * size_b[1] * size_b[2], 1e-6)
        return float(inter_vol / min(vol_a, vol_b))
