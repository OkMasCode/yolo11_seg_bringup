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
    max_radius: float = 0.0
    confidence_sum: float = 0.0


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
        # Recently lost map objects eligible for strict recovery.
        self.lost_map_objects: Dict[str, int] = {}

        self._next_map_id = 1

        # Confirmation/lifecycle policy.
        self.confirmation_min_hits = 15
        self.confirmation_time_window_sec = 1.5
        self.confirmation_min_age_sec = 0.6
        self.tentative_max_stale_sec = 1.5
        self.binding_ttl_sec = 4.0
        self.min_confidence_for_promotion = 0.55
        self.min_avg_confidence_for_promotion = 0.60
        self.min_class_vote_ratio_for_promotion = 0.70
        self.max_tentative_radius_for_promotion = 0.35
        # Input quality gates to reject low-quality detections early.
        self.min_input_confidence = 0.45
        self.min_detection_depth_m = 0.20
        self.max_detection_depth_m = 8.00
        self.min_box_extent_m = 0.03
        self.max_box_extent_m = 4.00
        self.min_box_volume_m3 = 0.0003
        self.max_box_volume_m3 = 25.0
        self.recovery_ttl_sec = 6.0
        self.max_recovery_distance = 0.45
        self.min_recovery_iou = 0.20
        self.min_recovery_size_similarity = 0.35
        # If upstream track IDs are unstable, allow strict same-class rebind
        # directly to active map objects to avoid duplicate proliferation.
        self.enable_strict_active_rebind = True
        self.min_occurrences_for_active_rebind = 4
        self.max_active_rebind_distance = 0.55
        self.min_active_rebind_iou = 0.12
        self.min_active_rebind_size_similarity = 0.25
        self.active_rebind_ambiguity_margin = 0.06

        # Adaptive gate parameters. The effective gate grows with object size.
        # Keep baseline-like minimum gate to avoid over-splitting into duplicates.
        self.base_distance_gate = 0.40
        self.size_distance_gain = 0.30
        self.max_distance_gate = 1.20

        # When class mismatch happens, we allow update only if geometry is very strong.
        self.class_mismatch_penalty = 0.10

        # Periodic dedup/fusion among confirmed map objects.
        self.merge_max_distance = 0.45

        # EMA coefficient for pose and confidence updates.
        self.pose_ema_alpha = 0.25
        self.confidence_ema_alpha = 0.20
        # For static objects, strongly damp updates after warmup to avoid centroid drift.
        self.pose_lock_after_occurrences = 6
        self.pose_locked_alpha = 0.05
        self.max_pose_step_locked = 0.08
        self.max_pose_step_unlocked_base = 0.18
        self.max_pose_step_unlocked_gain = 0.08
        self.min_same_class_gate = 0.55
        self.min_size_similarity_for_overlap = 0.25
        self.min_association_iou = 0.10
        self.merge_min_iou = 0.18
        self.merge_min_size_similarity = 0.45
        self.dedup_only_same_class = True
        self.enable_relaxed_large_object_dedup = True
        self.large_object_min_extent = 0.90
        self.relaxed_large_dedup_distance = 1.00
        self.relaxed_large_dedup_min_size_similarity = 0.30
        self.use_bbox_center_pose = False
        self.enable_online_dedup = True
        # Optional containment-based merge logic for same-class duplicates.
        self.enable_same_class_containment_merge = True
        self.containment_margin = 0.03
        self.min_containment_size_similarity = 0.15
        # Direct same-class containment merge: if existing object centroids fall
        # inside current detection bbox, reuse/fuse instead of creating duplicates.
        self.enable_detection_bbox_merge = True
        self.detection_bbox_merge_scale = 1.15
        self.detection_bbox_merge_margin = 0.05
        self.detection_bbox_merge_min_extent = 0.90
        self.detection_bbox_merge_min_volume = 0.20
        # Small-object protection: avoid merging nearby independent items.
        self.small_object_max_extent = 0.80
        self.small_object_assoc_min_iou = 0.22
        self.small_object_assoc_min_size_similarity = 0.35
        self.small_object_merge_max_distance = 0.40
        self.small_object_merge_min_iou = 0.30
        self.small_object_merge_min_size_similarity = 0.55

        # Run periodic duplicate fusion even when no new object is created.
        self._detections_since_merge = 0
        self.merge_every_n_detections = 20

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
        try:
            pose_cam = self._select_pose_cam(pose_in_camera, box_min, box_max)
            lookup_time = rclpy.time.Time.from_msg(detection_stamp)
            transform = self.tf_buffer.lookup_transform(
                fixed_frame,
                camera_frame,
                lookup_time,
                timeout=Duration(seconds=0.8),
            )
            pose_in_map = self.transform_point(pose_cam, transform)
        except TransformException as ex:
            self.node.get_logger().warning(f"[mapper_v2] Could not transform detection: {ex}")
            return False

        current_ns = self._stamp_to_ns(detection_stamp)
        self._prune_stale_state(current_ns)

        new_size = self._compute_box_size(box_min, box_max)
        img_vec = self._to_embedding(embeddings)

        # Drop implausible/low-quality detections before any association logic.
        if not self._passes_detection_quality_gate(confidence, pose_cam.z, new_size):
            return False

        # 1) Fast path: update through existing track->map binding.
        binding = self.track_bindings.get(tracker_id)
        if binding and binding.map_id in self.objects:
            obj = self.objects[binding.map_id]
            if self._is_valid_association(obj, pose_in_map, new_size, object_name):
                target_map_id = obj.map_id
                if self.enable_detection_bbox_merge:
                    merged_candidate_id = self._find_and_merge_contained_same_class_objects(
                        class_name=object_name,
                        pose_map=pose_in_map,
                        box_size=new_size,
                        current_ns=current_ns,
                    )
                    if merged_candidate_id is not None:
                        target_map_id = merged_candidate_id

                self._update_object(
                    map_id=target_map_id,
                    object_name=object_name,
                    detection_stamp=detection_stamp,
                    pose_in_camera=(pose_cam.x, pose_cam.y, pose_cam.z),
                    pose_in_map=pose_in_map,
                    box_size=new_size,
                    similarity=similarity,
                    confidence=confidence,
                    image_embedding=img_vec,
                    current_ns=current_ns,
                    source_track_id=tracker_id,
                )
                binding.map_id = target_map_id
                binding.last_seen_ns = current_ns
                binding.stable_hits += 1
                binding.misses = 0
                return False

            # Binding exists but geometry says it is no longer valid.
            binding.misses += 1
            if binding.misses >= 2:
                self._mark_binding_lost(binding)
                del self.track_bindings[tracker_id]

        # 2) Strict recovery path: only recover from recently lost objects.
        contained_candidate_id = self._find_and_merge_contained_same_class_objects(
            class_name=object_name,
            pose_map=pose_in_map,
            box_size=new_size,
            current_ns=current_ns,
        ) if self.enable_detection_bbox_merge else None

        if contained_candidate_id is not None:
            self.track_bindings[tracker_id] = TrackBinding(
                track_id=tracker_id,
                map_id=contained_candidate_id,
                first_seen_ns=current_ns,
                last_seen_ns=current_ns,
            )
            self._update_object(
                map_id=contained_candidate_id,
                object_name=object_name,
                detection_stamp=detection_stamp,
                pose_in_camera=(pose_cam.x, pose_cam.y, pose_cam.z),
                pose_in_map=pose_in_map,
                box_size=new_size,
                similarity=similarity,
                confidence=confidence,
                image_embedding=img_vec,
                current_ns=current_ns,
                source_track_id=tracker_id,
            )
            return False

        candidate_id = self._recover_lost_map_object(
            pose_map=pose_in_map,
            box_size=new_size,
            class_name=object_name,
            current_ns=current_ns,
        )
        if candidate_id is None and self.enable_strict_active_rebind:
            candidate_id = self._find_strict_active_candidate(
                pose_map=pose_in_map,
                box_size=new_size,
                class_name=object_name,
                current_ns=current_ns,
            )
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
                pose_in_camera=(pose_cam.x, pose_cam.y, pose_cam.z),
                pose_in_map=pose_in_map,
                box_size=new_size,
                similarity=similarity,
                confidence=confidence,
                image_embedding=img_vec,
                current_ns=current_ns,
                source_track_id=tracker_id,
            )
            return False

        # 3) No binding and no candidate: update tentative evidence for this track.
        created = self._update_tentative(
            object_name=object_name,
            tracker_id=tracker_id,
            pose_in_camera=pose_cam,
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
        if created and self.enable_online_dedup:
            self._merge_duplicates()

        self._detections_since_merge += 1
        if self.enable_online_dedup and self._detections_since_merge >= self.merge_every_n_detections:
            self._merge_duplicates()
            self._detections_since_merge = 0

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
                confidence_sum=max(float(confidence), 0.0),
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
                max_radius=0.0,
                confidence_sum=max(float(confidence), 0.0),
            )
            return False

        # Merge observation into tentative track.
        hits = tentative.hits + 1
        merged_pose = self._ema_tuple(tentative.pose_map, pose_in_map, self.pose_ema_alpha)
        merged_cam = self._ema_tuple(tentative.pose_cam, pose_cam, self.pose_ema_alpha)
        merged_size = self._ema_tuple(tentative.box_size, box_size, self.pose_ema_alpha)
        votes = dict(tentative.class_votes)
        votes[object_name] = votes.get(object_name, 0.0) + max(confidence, 0.01)
        max_radius = max(tentative.max_radius, dist)
        confidence_sum = tentative.confidence_sum + max(float(confidence), 0.0)

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
            max_radius=max_radius,
            confidence_sum=confidence_sum,
        )

        window_ns = int(self.confirmation_time_window_sec * 1e9)
        min_age_ns = int(self.confirmation_min_age_sec * 1e9)
        votes_total = max(sum(votes.values()), 1e-6)
        best_vote = max(votes.values())
        vote_ratio = float(best_vote / votes_total)
        avg_confidence = float(confidence_sum / max(hits, 1))
        promote_ok = (
            hits >= self.confirmation_min_hits
            and (current_ns - tentative.first_seen_ns) <= window_ns
            and (current_ns - tentative.first_seen_ns) >= min_age_ns
            and max(tentative.confidence_max, confidence) >= self.min_confidence_for_promotion
            and avg_confidence >= self.min_avg_confidence_for_promotion
            and vote_ratio >= self.min_class_vote_ratio_for_promotion
            and max_radius <= self.max_tentative_radius_for_promotion
        )
        if promote_ok:
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

        merged_pose = self._stable_pose_update(entry, pose_in_map)
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
        same_class_gate = max(gate * 1.25, self.min_same_class_gate)
        dist = self.euclidean_distance(pose_map, entry.pose_map)
        max_extent = max(max(entry.box_size), max(box_size))
        iou = self._aabb_iou(
            center_a=pose_map,
            size_a=box_size,
            center_b=entry.pose_map,
            size_b=entry.box_size,
        )
        size_similarity = self._size_similarity(entry.box_size, box_size)

        # If class differs, require stronger geometry evidence.
        if class_name != entry.current_name:
            return (
                dist <= (gate * (1.0 - self.class_mismatch_penalty))
                and iou >= self.min_association_iou
                and size_similarity >= self.min_size_similarity_for_overlap
            )

        # Same-class containment is a strong duplicate cue for static objects.
        if self.enable_same_class_containment_merge:
            if self._point_in_aabb(pose_map, entry.pose_map, entry.box_size, self.containment_margin):
                if size_similarity >= self.min_containment_size_similarity:
                    return True

        # Nearby small objects (chairs, plants, lamps) should not be merged by
        # centroid distance alone; require overlap + stronger size agreement.
        if max_extent <= self.small_object_max_extent:
            return (
                iou >= max(self.min_association_iou, self.small_object_assoc_min_iou)
                and size_similarity >= max(self.min_size_similarity_for_overlap, self.small_object_assoc_min_size_similarity)
            )

        # Same class can be associated by center proximity, but reject extreme size mismatch.
        if dist <= same_class_gate and size_similarity >= 0.12:
            return True

        return (
            iou >= self.min_association_iou
            and size_similarity >= self.min_size_similarity_for_overlap
        )

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
                same_class = a.current_name == b.current_name

                if self.dedup_only_same_class and not same_class:
                    continue

                # For static environments, same-class close objects should be merged more aggressively.
                class_extent_gate = 0.35 * (max(a.box_size) + max(b.box_size))
                adaptive_merge_gate = max(self.merge_max_distance, class_extent_gate)
                if dist > adaptive_merge_gate:
                    continue

                if not overlap:
                    continue

                iou = self._aabb_iou(a.pose_map, a.box_size, b.pose_map, b.box_size)
                size_similarity = self._size_similarity(a.box_size, b.box_size)
                max_extent = max(max(a.box_size), max(b.box_size))

                # Small-object protection: require tighter distance and much stronger
                # overlap/size evidence to merge nearby independent instances.
                if max_extent <= self.small_object_max_extent:
                    if dist > self.small_object_merge_max_distance:
                        continue
                    if iou < self.small_object_merge_min_iou:
                        continue
                    if size_similarity < self.small_object_merge_min_size_similarity:
                        continue

                strong_duplicate = (
                    iou >= self.merge_min_iou
                    and size_similarity >= self.merge_min_size_similarity
                )

                containment_duplicate = False
                if self.enable_same_class_containment_merge and same_class:
                    a_inside_b = self._point_in_aabb(a.pose_map, b.pose_map, b.box_size, self.containment_margin)
                    b_inside_a = self._point_in_aabb(b.pose_map, a.pose_map, a.box_size, self.containment_margin)
                    containment_duplicate = (
                        (a_inside_b or b_inside_a)
                        and size_similarity >= self.min_containment_size_similarity
                    )

                # For very large objects (e.g., beds/couches), map-frame jitter can make
                # AABB overlap weak even for the same physical item. Allow a distance+
                # size based fallback, but only for same-class large objects.
                large_object_relaxed_duplicate = False
                if self.enable_relaxed_large_object_dedup and same_class:
                    max_extent = max(max(a.box_size), max(b.box_size))
                    large_object_relaxed_duplicate = (
                        max_extent >= self.large_object_min_extent
                        and dist <= self.relaxed_large_dedup_distance
                        and size_similarity >= self.relaxed_large_dedup_min_size_similarity
                    )

                if not strong_duplicate and not large_object_relaxed_duplicate and not containment_duplicate:
                    continue

                # Keep the older map object as canonical and absorb the newer one.
                keep_id, drop_id = (a_id, b_id) if a.first_seen_ns <= b.first_seen_ns else (b_id, a_id)
                self._fuse_objects(keep_id, drop_id)
                merged_pairs.append((keep_id, drop_id))

        if merged_pairs:
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
        recovery_ttl_ns = int(self.recovery_ttl_sec * 1e9)

        for track_id, tentative in list(self.tentative_tracks.items()):
            if current_ns - tentative.last_seen_ns > stale_tentative_ns:
                del self.tentative_tracks[track_id]

        for track_id, binding in list(self.track_bindings.items()):
            if current_ns - binding.last_seen_ns > stale_binding_ns:
                self._mark_binding_lost(binding)
                del self.track_bindings[track_id]

        for map_id, lost_ns in list(self.lost_map_objects.items()):
            if current_ns - lost_ns > recovery_ttl_ns:
                del self.lost_map_objects[map_id]

    def _mark_binding_lost(self, binding: TrackBinding) -> None:
        """Remember when a map object lost its active track binding."""
        if binding.map_id in self.objects:
            self.lost_map_objects[binding.map_id] = int(binding.last_seen_ns)

    def _recover_lost_map_object(self, pose_map, box_size, class_name: str, current_ns: int) -> Optional[str]:
        """Recover only from recent lost objects using strict geometry and class agreement."""
        recovery_ttl_ns = int(self.recovery_ttl_sec * 1e9)
        best_id = None
        best_score = float('inf')
        second_best = float('inf')

        for map_id, lost_ns in self.lost_map_objects.items():
            if (current_ns - lost_ns) > recovery_ttl_ns:
                continue

            entry = self.objects.get(map_id)
            if entry is None:
                continue
            if entry.current_name != class_name:
                continue

            dist = self.euclidean_distance(pose_map, entry.pose_map)
            if dist > self.max_recovery_distance:
                continue

            iou = self._aabb_iou(pose_map, box_size, entry.pose_map, entry.box_size)
            size_similarity = self._size_similarity(entry.box_size, box_size)
            if iou < self.min_recovery_iou:
                continue
            if size_similarity < self.min_recovery_size_similarity:
                continue

            score = 0.70 * dist + 0.20 * (1.0 - iou) + 0.10 * (1.0 - size_similarity)
            if score < best_score:
                second_best = best_score
                best_score = score
                best_id = map_id
            elif score < second_best:
                second_best = score

        if best_id is None:
            return None

        # Reject ambiguous recovery choices in clutter.
        if (second_best - best_score) < 0.05:
            return None

        self.lost_map_objects.pop(best_id, None)
        return best_id

    def _find_strict_active_candidate(self, pose_map, box_size, class_name: str, current_ns: int) -> Optional[str]:
        """Find a strong same-class active candidate to prevent duplicate object creation."""
        best_id = None
        best_score = float('inf')
        second_best = float('inf')

        for map_id, entry in self.objects.items():
            if entry.current_name != class_name:
                continue
            if entry.occurrences < self.min_occurrences_for_active_rebind:
                continue

            # Ignore very stale objects to avoid snapping to old map content.
            if (current_ns - entry.last_seen_ns) > int(self.recovery_ttl_sec * 1e9):
                continue

            gate = min(self.max_active_rebind_distance, self._dynamic_gate(entry.box_size, box_size))
            dist = self.euclidean_distance(pose_map, entry.pose_map)
            if dist > gate:
                continue

            iou = self._aabb_iou(pose_map, box_size, entry.pose_map, entry.box_size)
            size_similarity = self._size_similarity(entry.box_size, box_size)
            if size_similarity < self.min_active_rebind_size_similarity:
                continue

            # Require either overlap evidence or very close centroid agreement.
            very_close = dist <= min(0.20, 0.5 * gate)
            if iou < self.min_active_rebind_iou and not very_close:
                continue

            score = (
                0.60 * (dist / max(gate, 1e-6))
                + 0.25 * (1.0 - min(iou, 1.0))
                + 0.15 * (1.0 - size_similarity)
            )
            if score < best_score:
                second_best = best_score
                best_score = score
                best_id = map_id
            elif score < second_best:
                second_best = score

        if best_id is None:
            return None

        # Reject ambiguous matches in dense same-class clutter.
        if (second_best - best_score) < self.active_rebind_ambiguity_margin:
            return None

        return best_id

    def _find_and_merge_contained_same_class_objects(
        self,
        class_name: str,
        pose_map: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        current_ns: int,
    ) -> Optional[str]:
        """
        Return a same-class candidate whose centroid is inside the current detection bbox.

        If multiple same-class objects are inside the detection bbox, fuse them into one
        canonical object to prevent duplicate proliferation.
        """
        # Protect small nearby objects (e.g. chairs, plants, lamps): containment-based
        # merging is reserved for detections large enough to represent big furniture.
        det_extent = max(float(box_size[0]), float(box_size[1]), float(box_size[2]))
        det_volume = max(float(box_size[0] * box_size[1] * box_size[2]), 1e-6)
        if det_extent < self.detection_bbox_merge_min_extent:
            return None
        if det_volume < self.detection_bbox_merge_min_volume:
            return None

        scaled_size = tuple(max(float(s) * self.detection_bbox_merge_scale, 0.01) for s in box_size)

        candidate_ids = []
        for map_id, entry in self.objects.items():
            if entry.current_name != class_name:
                continue

            if self._point_in_aabb(
                point=entry.pose_map,
                box_center=pose_map,
                box_size=scaled_size,
                margin=self.detection_bbox_merge_margin,
            ):
                candidate_ids.append(map_id)

        if not candidate_ids:
            return None

        # Keep the oldest object ID stable as canonical.
        keep_id = min(candidate_ids, key=lambda mid: self.objects[mid].first_seen_ns)

        # Fuse all same-class centroids captured by this detection bbox.
        for drop_id in candidate_ids:
            if drop_id == keep_id:
                continue
            if drop_id in self.objects and keep_id in self.objects:
                self._fuse_objects(keep_id, drop_id)

        # Refresh object recency so subsequent active rebind logic treats it as active.
        if keep_id in self.objects:
            kept = self.objects[keep_id]
            kept.last_seen_ns = max(kept.last_seen_ns, current_ns)

        return keep_id

    def _new_map_id(self) -> str:
        map_id = f"map_obj_{self._next_map_id:06d}"
        self._next_map_id += 1
        return map_id

    # ----------------- Serialization ------------------ #

    def export_to_json(self, directory_path: str, file: str = 'map_v2.json') -> None:
        """Export map objects using the same JSON schema as the baseline mapper."""
        os.makedirs(directory_path, exist_ok=True)
        path = os.path.join(directory_path, file)

        export_data = {}

        for map_id, obj in self.objects.items():
            similarity_value = float(obj.similarity) if obj.similarity is not None else None
            export_data[map_id] = {
                'name': obj.current_name,
                'frame': obj.frame,
                'timestamp': {'sec': obj.timestamp.sec, 'nanosec': obj.timestamp.nanosec},
                'pose_map': {'x': float(obj.pose_map[0]), 'y': float(obj.pose_map[1]), 'z': float(obj.pose_map[2])},
                'occurrences': int(obj.occurrences),
                'similarity': similarity_value,
                'image_embedding': obj.image_embedding.tolist() if obj.image_embedding is not None else None,
                'confidence': float(obj.confidence_ema) if obj.confidence_ema is not None else 0.0,
            }

        with open(path, 'w') as json_file:
            json.dump(export_data, json_file, indent=4)

    def load_from_json(self, directory_path: str, file: str = 'map_v2.json') -> None:
        """Load map data from baseline-compatible JSON schema."""
        path = os.path.join(directory_path, file)
        if not os.path.exists(path):
            self.node.get_logger().info(f"[mapper_v2] No map file found at {path}, starting empty")
            return

        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)

            if not isinstance(data, dict):
                self.node.get_logger().warning(f"[mapper_v2] Unexpected map payload type: {type(data).__name__}")
                return

            loaded_objects = data
            self.objects = {}

            max_numeric_id = 0
            for map_id, obj_data in loaded_objects.items():
                ts = Time(sec=int(obj_data['timestamp']['sec']), nanosec=int(obj_data['timestamp']['nanosec']))
                pose_map = (
                    float(obj_data['pose_map']['x']),
                    float(obj_data['pose_map']['y']),
                    float(obj_data['pose_map']['z']),
                )
                # Baseline exports only pose_map, so reuse it for pose_cam on reload.
                pose_cam = pose_map
                box_size = (0.10, 0.10, 0.10)

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
                    first_seen_ns=self._stamp_to_ns(ts),
                    last_seen_ns=self._stamp_to_ns(ts),
                    current_name=obj_data.get('name', 'unknown'),
                    class_votes={obj_data.get('name', 'unknown'): float(obj_data.get('confidence', 0.0) or 1.0)},
                    similarity=float(obj_data.get('similarity', 0.0)),
                    confidence_ema=float(obj_data.get('confidence', 0.0)),
                    image_embedding=image_embedding,
                    source_track_id=None,
                )

                if map_id.startswith('map_obj_'):
                    try:
                        max_numeric_id = max(max_numeric_id, int(map_id.split('_')[-1]))
                    except ValueError:
                        pass

            self._next_map_id = max_numeric_id + 1
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

    def _compute_box_size(self, box_min, box_max) -> Tuple[float, float, float]:
        if box_min is None or box_max is None:
            return (0.10, 0.10, 0.10)
        return (
            max(abs(box_max[0] - box_min[0]), 0.01),
            max(abs(box_max[1] - box_min[1]), 0.01),
            max(abs(box_max[2] - box_min[2]), 0.01),
        )

    def _passes_detection_quality_gate(
        self,
        confidence: float,
        depth_m: float,
        box_size: Tuple[float, float, float],
    ) -> bool:
        """Return True when a detection is reliable enough to enter mapper logic."""
        if float(confidence) < self.min_input_confidence:
            return False

        if not math.isfinite(float(depth_m)):
            return False
        if float(depth_m) < self.min_detection_depth_m or float(depth_m) > self.max_detection_depth_m:
            return False

        extent = max(float(box_size[0]), float(box_size[1]), float(box_size[2]))
        if extent < self.min_box_extent_m or extent > self.max_box_extent_m:
            return False

        volume = float(box_size[0] * box_size[1] * box_size[2])
        if volume < self.min_box_volume_m3 or volume > self.max_box_volume_m3:
            return False

        return True

    def _select_pose_cam(self, pose_in_camera, box_min, box_max) -> Vector3:
        """
        Choose camera-frame position used for mapping.

        Prefer bbox center when available to reduce side-biased centroids that come
        from partial point-cloud visibility.
        """
        if self.use_bbox_center_pose and box_min is not None and box_max is not None:
            return Vector3(
                x=0.5 * float(box_min[0] + box_max[0]),
                y=0.5 * float(box_min[1] + box_max[1]),
                z=0.5 * float(box_min[2] + box_max[2]),
            )
        return Vector3(
            x=float(pose_in_camera.x),
            y=float(pose_in_camera.y),
            z=float(pose_in_camera.z),
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

    def _stable_pose_update(self, entry: MapObject, new_pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Update map pose under static-world assumption.

        Early observations can move faster to converge, then updates are heavily damped
        and step-clamped so camera motion does not drag object centroids around.
        """
        if entry.occurrences < self.pose_lock_after_occurrences:
            alpha = self.pose_ema_alpha
            max_step = self.max_pose_step_unlocked_base + self.max_pose_step_unlocked_gain * max(entry.box_size)
        else:
            alpha = self.pose_locked_alpha
            max_step = self.max_pose_step_locked

        blended = self._ema_tuple(entry.pose_map, new_pose, alpha)
        dx = blended[0] - entry.pose_map[0]
        dy = blended[1] - entry.pose_map[1]
        dz = blended[2] - entry.pose_map[2]
        step = math.sqrt(dx * dx + dy * dy + dz * dz)
        if step <= max_step:
            return blended

        scale = max_step / max(step, 1e-6)
        return (
            entry.pose_map[0] + dx * scale,
            entry.pose_map[1] + dy * scale,
            entry.pose_map[2] + dz * scale,
        )

    def _size_similarity(self, size_a: Tuple[float, float, float], size_b: Tuple[float, float, float]) -> float:
        """Return size similarity in [0, 1] from volume ratio (1 is most similar)."""
        va = max(size_a[0] * size_a[1] * size_a[2], 1e-6)
        vb = max(size_b[0] * size_b[1] * size_b[2], 1e-6)
        ratio = min(va, vb) / max(va, vb)
        return float(max(0.0, min(1.0, ratio)))

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

    def _aabb_iou(self, center_a, size_a, center_b, size_b) -> float:
        """Axis-aligned 3D IoU for robust association and dedup decisions."""
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
        union = max(vol_a + vol_b - inter_vol, 1e-6)
        return float(inter_vol / union)

    def _point_in_aabb(self, point, box_center, box_size, margin: float = 0.0) -> bool:
        """Check if a 3D point is inside an AABB (optionally expanded by margin)."""
        half = np.array(box_size, dtype=np.float64) / 2.0 + float(max(margin, 0.0))
        p = np.array(point, dtype=np.float64)
        c = np.array(box_center, dtype=np.float64)
        delta = np.abs(p - c)
        return bool(np.all(delta <= half))
