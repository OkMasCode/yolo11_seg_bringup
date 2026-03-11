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
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_conf_sums: Dict[str, float] = field(default_factory=dict)
    similarity: float = 0.0
    confidence_ema: float = 0.0
    image_embedding: Optional[np.ndarray] = None
    embedding_confidence_max: float = -1.0
    source_track_id: Optional[str] = None


@dataclass
class TentativeTrack:
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
    confidence_max: float = 0.0
    confidence_sum: float = 0.0
    image_embedding: Optional[np.ndarray] = None
    embedding_confidence_max: float = -1.0


def quaternion_matrix(quaternion):
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


class SemanticObjectMapV3:
    """
    Simplified semantic mapper.

    Main behavior:
    1) Reject low-quality detections early.
    2) Associate with existing map object (same class + geometry).
    3) For large detections, merge same-class centroids inside detection bbox.
    4) Use tentative tracks to avoid one-frame false positives.
    """

    def __init__(self, tf_buffer: Buffer, node: Node):
        self.tf_buffer = tf_buffer
        self.node = node

        self.objects: Dict[str, MapObject] = {}
        self.tentative_tracks: Dict[str, TentativeTrack] = {}
        self.track_to_map: Dict[str, str] = {}
        self.track_last_seen_ns: Dict[str, int] = {}
        self._next_map_id = 1

        # Detection quality gates.
        self.min_input_confidence = 0.55
        self.min_detection_depth_m = 0.25
        self.max_detection_depth_m = 6.0
        self.min_box_extent_m = 0.03
        self.max_box_extent_m = 3.5
        self.min_box_volume_m3 = 0.0005
        self.max_box_volume_m3 = 15.0

        # Association.
        self.base_distance_gate = 0.40
        self.size_distance_gain = 0.30
        self.max_distance_gate = 1.20
        self.min_association_iou = 0.12
        self.class_mismatch_penalty = 0.15
        self.min_cross_class_iou = 0.20

        # Large object containment merge.
        self.enable_detection_bbox_merge = True
        self.detection_bbox_merge_scale = 1.10
        self.detection_bbox_merge_margin = 0.03
        self.detection_bbox_merge_min_extent = 1.00

        # Small-object protection.
        self.small_object_max_extent = 1.00
        self.small_object_assoc_min_iou = 0.30

        # Tentative confirmation (false-positive suppression).
        self.confirmation_min_hits = 6
        self.confirmation_time_window_sec = 2.5
        self.confirmation_min_age_sec = 0.8
        self.min_confidence_for_promotion = 0.50
        self.min_avg_confidence_for_promotion = 0.55
        self.tentative_max_stale_sec = 2.0

        # Binding lifecycle.
        self.binding_ttl_sec = 4.0

        # Smoothing.
        self.pose_ema_alpha = 0.25
        self.size_ema_alpha = 0.20
        self.confidence_ema_alpha = 0.20

        # Class consensus (count + confidence) with hysteresis against rapid flips.
        self.class_count_weight = 1.0
        self.class_confidence_weight = 2.0
        self.class_switch_margin = 0.75
        self.min_class_votes_to_lock = 4

    def _normalize_embedding(self, embedding) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return None
        norm = float(np.linalg.norm(vec))
        if not math.isfinite(norm) or norm <= 1e-12:
            return None
        return vec / norm

    def _fuse_embeddings_running_avg(
        self,
        current_embedding,
        current_count: int,
        new_embedding,
        new_count: int = 1,
    ) -> Optional[np.ndarray]:
        cur = self._normalize_embedding(current_embedding)
        new = self._normalize_embedding(new_embedding)

        if cur is None:
            return new
        if new is None:
            return cur

        cur_w = max(int(current_count), 1)
        new_w = max(int(new_count), 1)
        fused = (cur * float(cur_w) + new * float(new_w)) / float(cur_w + new_w)
        return self._normalize_embedding(fused)

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
        try:
            # Convert raw detection into camera-frame point and transform it into map frame.
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
            self.node.get_logger().warning(f"[mapper_v3] Could not transform detection: {ex}")
            return False

        # Build map-frame AABB by transforming the 8 camera-frame bbox corners.
        bbox_center_map, new_size = self._transform_bbox_to_map_aabb(box_min, box_max, transform)
        if bbox_center_map is not None:
            pose_in_map = bbox_center_map
        else:
            new_size = self._compute_box_size(box_min, box_max)

        # Keep state fresh before any association decisions.
        current_ns = self._stamp_to_ns(detection_stamp)
        self._prune_stale_state(current_ns)
        img_vec = self._to_embedding(embeddings)

        if not self._passes_detection_quality_gate(confidence, pose_cam.z, new_size):
            return False

        # First try the tracker's previous map binding (fast path, preserves identity).
        bound_map_id = self.track_to_map.get(tracker_id)
        if bound_map_id in self.objects:
            obj = self.objects[bound_map_id]
            if self._is_valid_association(obj, pose_in_map, new_size, object_name):
                target_id = bound_map_id
                if self.enable_detection_bbox_merge:
                    merged_id = self._merge_contained_same_class(object_name, pose_in_map, new_size)
                    if merged_id is not None:
                        target_id = merged_id
                self._update_object(
                    map_id=target_id,
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
                self.track_to_map[tracker_id] = target_id
                self.track_last_seen_ns[tracker_id] = current_ns
                return False
            # Binding is stale/invalid for current geometry; force re-association.
            self.track_to_map.pop(tracker_id, None)
            self.track_last_seen_ns.pop(tracker_id, None)

        # Optional merge for large detections that geometrically contain same-class objects.
        if self.enable_detection_bbox_merge:
            merged_id = self._merge_contained_same_class(object_name, pose_in_map, new_size)
            if merged_id is not None:
                self._update_object(
                    map_id=merged_id,
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
                self.track_to_map[tracker_id] = merged_id
                self.track_last_seen_ns[tracker_id] = current_ns
                return False

            # Global nearest-candidate lookup when no binding/merge match is available.
        candidate_id = self._find_best_candidate(object_name, pose_in_map, new_size)
        if candidate_id is not None:
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
            self.track_to_map[tracker_id] = candidate_id
            self.track_last_seen_ns[tracker_id] = current_ns
            return False

        # No confirmed object matched; update or create a tentative track.
        created = self._update_tentative(
            object_name=object_name,
            tracker_id=tracker_id,
            pose_in_camera=pose_cam,
            pose_in_map=pose_in_map,
            box_size=new_size,
            detection_stamp=detection_stamp,
            confidence=confidence,
            image_embedding=img_vec,
            current_ns=current_ns,
            frame=camera_frame,
        )
        return created

    def _update_tentative(
        self,
        object_name: str,
        tracker_id: str,
        pose_in_camera,
        pose_in_map,
        box_size,
        detection_stamp,
        confidence: float,
        image_embedding,
        current_ns: int,
        frame: str,
    ) -> bool:
        pose_cam = (pose_in_camera.x, pose_in_camera.y, pose_in_camera.z)
        track = self.tentative_tracks.get(tracker_id)

        if track is None:
            # Start a new tentative track; require more evidence before map insertion.
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
                confidence_max=float(confidence),
                confidence_sum=max(float(confidence), 0.0),
                image_embedding=self._normalize_embedding(image_embedding),
                embedding_confidence_max=(float(confidence) if image_embedding is not None else -1.0),
            )
            return False

        # Keep tentative identity stable and simple.
        if object_name != track.class_name:
            # Reset track if class flips while still tentative.
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
                confidence_max=float(confidence),
                confidence_sum=max(float(confidence), 0.0),
                image_embedding=self._normalize_embedding(image_embedding),
                embedding_confidence_max=(float(confidence) if image_embedding is not None else -1.0),
            )
            return False

        # Smooth tentative pose/size over time to reduce frame-to-frame jitter.
        merged_pose = self._ema_tuple(track.pose_map, pose_in_map, self.pose_ema_alpha)
        merged_cam = self._ema_tuple(track.pose_cam, pose_cam, self.pose_ema_alpha)
        merged_size = self._ema_tuple(track.box_size, box_size, self.size_ema_alpha)
        hits = track.hits + 1
        conf_sum = track.confidence_sum + max(float(confidence), 0.0)

        # Keep a running average of embeddings and re-normalize after each update.
        fused_embedding = self._fuse_embeddings_running_avg(
            current_embedding=track.image_embedding,
            current_count=track.hits,
            new_embedding=image_embedding,
            new_count=1,
        )

        best_embedding_conf = float(track.embedding_confidence_max)
        if image_embedding is not None:
            best_embedding_conf = max(best_embedding_conf, float(confidence))

        self.tentative_tracks[tracker_id] = TentativeTrack(
            track_id=tracker_id,
            frame=frame,
            timestamp=detection_stamp,
            pose_cam=merged_cam,
            pose_map=merged_pose,
            box_size=merged_size,
            hits=hits,
            first_seen_ns=track.first_seen_ns,
            last_seen_ns=current_ns,
            class_name=track.class_name,
            confidence_max=max(track.confidence_max, float(confidence)),
            confidence_sum=conf_sum,
            image_embedding=fused_embedding,
            embedding_confidence_max=best_embedding_conf,
        )

        age_ns = current_ns - track.first_seen_ns
        window_ns = int(self.confirmation_time_window_sec * 1e9)
        min_age_ns = int(self.confirmation_min_age_sec * 1e9)
        avg_conf = conf_sum / max(hits, 1)

        # Promote tentative -> confirmed map object only with enough stable evidence.
        promote_ok = (
            hits >= self.confirmation_min_hits
            and age_ns >= min_age_ns
            and age_ns <= window_ns
            and max(track.confidence_max, float(confidence)) >= self.min_confidence_for_promotion
            and avg_conf >= self.min_avg_confidence_for_promotion
        )

        if not promote_ok:
            return False

        # Confirmation passed: convert tentative track into a persistent map object.
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
            class_votes={promoted.class_name: promoted.confidence_sum},
            class_counts={promoted.class_name: int(promoted.hits)},
            class_conf_sums={promoted.class_name: float(promoted.confidence_sum)},
            similarity=0.0,
            confidence_ema=float(promoted.confidence_max),
            image_embedding=promoted.image_embedding,
            embedding_confidence_max=float(promoted.embedding_confidence_max),
            source_track_id=tracker_id,
        )
        self.track_to_map[tracker_id] = map_id
        self.track_last_seen_ns[tracker_id] = current_ns
        return True

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
        entry = self.objects[map_id]

        # Exponential moving average for stable map geometry.
        merged_pose = self._ema_tuple(entry.pose_map, pose_in_map, self.pose_ema_alpha)
        merged_cam = self._ema_tuple(entry.pose_cam, pose_in_camera, self.pose_ema_alpha)
        merged_size = self._ema_tuple(entry.box_size, box_size, self.size_ema_alpha)

        # Accumulate label evidence by both count and confidence.
        class_votes = dict(entry.class_votes)
        class_votes[object_name] = class_votes.get(object_name, 0.0) + max(float(confidence), 0.01)

        class_counts = dict(entry.class_counts)
        class_counts[object_name] = class_counts.get(object_name, 0) + 1

        class_conf_sums = dict(entry.class_conf_sums)
        class_conf_sums[object_name] = class_conf_sums.get(object_name, 0.0) + max(float(confidence), 0.01)

        best_class = self._choose_consensus_class(
            class_counts=class_counts,
            class_conf_sums=class_conf_sums,
            current_name=entry.current_name,
        )

        # Smooth confidence so downstream consumers see less noisy confidence traces.
        conf_ema = (1.0 - self.confidence_ema_alpha) * entry.confidence_ema + self.confidence_ema_alpha * float(confidence)

        embedding = self._fuse_embeddings_running_avg(
            current_embedding=entry.image_embedding,
            current_count=entry.occurrences,
            new_embedding=image_embedding,
            new_count=1,
        )

        embedding_confidence_max = float(entry.embedding_confidence_max)
        if image_embedding is not None:
            embedding_confidence_max = max(embedding_confidence_max, float(confidence))

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
            class_counts=class_counts,
            class_conf_sums=class_conf_sums,
            similarity=max(entry.similarity, float(similarity)),
            confidence_ema=conf_ema,
            image_embedding=embedding,
            embedding_confidence_max=embedding_confidence_max,
            source_track_id=source_track_id,
        )

    def _is_valid_association(self, entry: MapObject, pose_map, box_size, class_name: str) -> bool:
        dist = self.euclidean_distance(pose_map, entry.pose_map)
        iou = self._aabb_iou(pose_map, box_size, entry.pose_map, entry.box_size)
        extent = max(max(entry.box_size), max(box_size))
        gate = self._dynamic_gate(entry.box_size, box_size)
        same_class = class_name == entry.current_name

        if extent <= self.small_object_max_extent:
            min_iou = max(self.min_association_iou, self.small_object_assoc_min_iou)
            if not same_class:
                min_iou = max(min_iou, self.min_cross_class_iou)
            return iou >= min_iou

        if not same_class:
            # Allow relabeling only when geometry is strong enough.
            return (
                iou >= self.min_cross_class_iou
                and dist <= (1.0 - self.class_mismatch_penalty) * gate
            )

        return dist <= gate or iou >= self.min_association_iou

    def _find_best_candidate(
        self,
        class_name: str,
        pose_map: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
    ) -> Optional[str]:
        best_id = None
        best_score = float('inf')

        for map_id, entry in self.objects.items():
            dist = self.euclidean_distance(pose_map, entry.pose_map)
            gate = self._dynamic_gate(entry.box_size, box_size)
            if dist > gate:
                continue

            iou = self._aabb_iou(pose_map, box_size, entry.pose_map, entry.box_size)
            if entry.current_name != class_name and iou < self.min_cross_class_iou:
                continue

            mismatch = 1.0 if entry.current_name != class_name else 0.0
            score = 0.65 * (dist / max(gate, 1e-6)) + 0.30 * (1.0 - iou) + 0.05 * mismatch
            if score < best_score:
                best_score = score
                best_id = map_id

        return best_id

    def _merge_contained_same_class(
        self,
        class_name: str,
        pose_map: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
    ) -> Optional[str]:
        extent = max(float(box_size[0]), float(box_size[1]), float(box_size[2]))
        if extent < self.detection_bbox_merge_min_extent:
            return None

        scaled_size = tuple(max(float(s) * self.detection_bbox_merge_scale, 0.01) for s in box_size)

        candidates = []
        for map_id, entry in self.objects.items():
            if entry.current_name != class_name:
                continue
            if self._point_in_aabb(entry.pose_map, pose_map, scaled_size, self.detection_bbox_merge_margin):
                candidates.append(map_id)

        if not candidates:
            return None

        keep_id = min(candidates, key=lambda mid: self.objects[mid].first_seen_ns)
        for drop_id in candidates:
            if drop_id != keep_id and drop_id in self.objects and keep_id in self.objects:
                self._fuse_objects(keep_id, drop_id)
        return keep_id

    def _fuse_objects(self, keep_id: str, drop_id: str) -> None:
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

        class_counts = dict(keep.class_counts)
        for name, count in drop.class_counts.items():
            class_counts[name] = class_counts.get(name, 0) + int(count)

        class_conf_sums = dict(keep.class_conf_sums)
        for name, conf_sum in drop.class_conf_sums.items():
            class_conf_sums[name] = class_conf_sums.get(name, 0.0) + float(conf_sum)

        best_class = self._choose_consensus_class(
            class_counts=class_counts,
            class_conf_sums=class_conf_sums,
            current_name=keep.current_name,
        )

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
            class_counts=class_counts,
            class_conf_sums=class_conf_sums,
            similarity=max(keep.similarity, drop.similarity),
            confidence_ema=max(keep.confidence_ema, drop.confidence_ema),
            image_embedding=self._fuse_embeddings_running_avg(
                current_embedding=keep.image_embedding,
                current_count=keep.occurrences,
                new_embedding=drop.image_embedding,
                new_count=drop.occurrences,
            ),
            embedding_confidence_max=max(float(keep.embedding_confidence_max), float(drop.embedding_confidence_max)),
            source_track_id=keep.source_track_id,
        )
        del self.objects[drop_id]

        for track_id, mapped_id in list(self.track_to_map.items()):
            if mapped_id == drop_id:
                self.track_to_map[track_id] = keep_id

    def _prune_stale_state(self, current_ns: int) -> None:
        stale_tentative_ns = int(self.tentative_max_stale_sec * 1e9)
        stale_binding_ns = int(self.binding_ttl_sec * 1e9)

        # Drop tentative tracks that stopped receiving observations.
        for track_id, t in list(self.tentative_tracks.items()):
            if current_ns - t.last_seen_ns > stale_tentative_ns:
                del self.tentative_tracks[track_id]

        # Drop old tracker bindings to avoid reconnecting stale IDs to map objects.
        for track_id, last_seen in list(self.track_last_seen_ns.items()):
            if current_ns - last_seen > stale_binding_ns:
                self.track_last_seen_ns.pop(track_id, None)
                self.track_to_map.pop(track_id, None)

    def _new_map_id(self) -> str:
        map_id = f"map_obj_{self._next_map_id:06d}"
        self._next_map_id += 1
        return map_id

    def export_to_json(self, directory_path: str, file: str = 'map_v3.json') -> None:
        os.makedirs(directory_path, exist_ok=True)
        path = os.path.join(directory_path, file)

        export_data = {}
        for map_id, obj in self.objects.items():
            hx = float(obj.box_size[0]) * 0.5
            hy = float(obj.box_size[1]) * 0.5
            hz = float(obj.box_size[2]) * 0.5
            cx = float(obj.pose_map[0])
            cy = float(obj.pose_map[1])
            cz = float(obj.pose_map[2])
            bbox_min = {'x': cx - hx, 'y': cy - hy, 'z': cz - hz}
            bbox_max = {'x': cx + hx, 'y': cy + hy, 'z': cz + hz}
            export_data[map_id] = {
                'name': obj.current_name,
                'frame': obj.frame,
                'timestamp': {'sec': obj.timestamp.sec, 'nanosec': obj.timestamp.nanosec},
                'pose_map': {'x': float(obj.pose_map[0]), 'y': float(obj.pose_map[1]), 'z': float(obj.pose_map[2])},
                'bbox_type': 'aabb',
                'box_size': {'x': float(obj.box_size[0]), 'y': float(obj.box_size[1]), 'z': float(obj.box_size[2])},
                'bbox_min': bbox_min,
                'bbox_max': bbox_max,
                'bbox_corners': [
                    {'x': bbox_min['x'], 'y': bbox_min['y'], 'z': bbox_min['z']},
                    {'x': bbox_min['x'], 'y': bbox_min['y'], 'z': bbox_max['z']},
                    {'x': bbox_min['x'], 'y': bbox_max['y'], 'z': bbox_min['z']},
                    {'x': bbox_min['x'], 'y': bbox_max['y'], 'z': bbox_max['z']},
                    {'x': bbox_max['x'], 'y': bbox_min['y'], 'z': bbox_min['z']},
                    {'x': bbox_max['x'], 'y': bbox_min['y'], 'z': bbox_max['z']},
                    {'x': bbox_max['x'], 'y': bbox_max['y'], 'z': bbox_min['z']},
                    {'x': bbox_max['x'], 'y': bbox_max['y'], 'z': bbox_max['z']},
                ],
                'occurrences': int(obj.occurrences),
                'similarity': float(obj.similarity),
                'image_embedding': obj.image_embedding.tolist() if obj.image_embedding is not None else None,
                'embedding_confidence': float(obj.embedding_confidence_max),
                'confidence': float(obj.confidence_ema),
            }

        with open(path, 'w') as json_file:
            json.dump(export_data, json_file, indent=4)

    def load_from_json(self, directory_path: str, file: str = 'map_v3.json') -> None:
        path = os.path.join(directory_path, file)
        if not os.path.exists(path):
            self.node.get_logger().info(f"[mapper_v3] No map file found at {path}, starting empty")
            return

        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
            if not isinstance(data, dict):
                return

            self.objects = {}
            max_numeric_id = 0

            for map_id, obj_data in data.items():
                ts = Time(sec=int(obj_data['timestamp']['sec']), nanosec=int(obj_data['timestamp']['nanosec']))
                pose_map = (
                    float(obj_data['pose_map']['x']),
                    float(obj_data['pose_map']['y']),
                    float(obj_data['pose_map']['z']),
                )
                raw_box_size = obj_data.get('box_size')
                if isinstance(raw_box_size, dict):
                    box_size = (
                        max(float(raw_box_size.get('x', 0.10)), 0.01),
                        max(float(raw_box_size.get('y', 0.10)), 0.01),
                        max(float(raw_box_size.get('z', 0.10)), 0.01),
                    )
                elif isinstance(raw_box_size, (list, tuple)) and len(raw_box_size) >= 3:
                    box_size = (
                        max(float(raw_box_size[0]), 0.01),
                        max(float(raw_box_size[1]), 0.01),
                        max(float(raw_box_size[2]), 0.01),
                    )
                else:
                    box_size = (0.10, 0.10, 0.10)
                img = obj_data.get('image_embedding')
                image_embedding = self._normalize_embedding(np.array(img, dtype=np.float32) if img else None)

                self.objects[map_id] = MapObject(
                    map_id=map_id,
                    frame=obj_data.get('frame', 'map'),
                    timestamp=ts,
                    pose_cam=pose_map,
                    pose_map=pose_map,
                    box_size=box_size,
                    occurrences=int(obj_data.get('occurrences', 1)),
                    first_seen_ns=self._stamp_to_ns(ts),
                    last_seen_ns=self._stamp_to_ns(ts),
                    current_name=obj_data.get('name', 'unknown'),
                    class_votes={obj_data.get('name', 'unknown'): float(obj_data.get('confidence', 0.0) or 1.0)},
                    class_counts={obj_data.get('name', 'unknown'): int(obj_data.get('occurrences', 1))},
                    class_conf_sums={
                        obj_data.get('name', 'unknown'): float(
                            (obj_data.get('confidence', 0.0) or 1.0) * max(int(obj_data.get('occurrences', 1)), 1)
                        )
                    },
                    similarity=float(obj_data.get('similarity', 0.0)),
                    confidence_ema=float(obj_data.get('confidence', 0.0)),
                    image_embedding=image_embedding,
                    embedding_confidence_max=float(obj_data.get('embedding_confidence', -1.0)),
                    source_track_id=None,
                )

                if map_id.startswith('map_obj_'):
                    try:
                        max_numeric_id = max(max_numeric_id, int(map_id.split('_')[-1]))
                    except ValueError:
                        pass

            self._next_map_id = max_numeric_id + 1
            self.node.get_logger().info(f"[mapper_v3] Loaded {len(self.objects)} objects from {path}")

        except Exception as ex:
            self.node.get_logger().error(f"[mapper_v3] Error loading map from {path}: {ex}")

    def transform_point(self, point, transform: TransformStamped) -> Tuple[float, float, float]:
        q = transform.transform.rotation
        t = transform.transform.translation
        rot_matrix = quaternion_matrix([q.w, q.x, q.y, q.z])[:3, :3]
        point_vec = np.array([[point.x], [point.y], [point.z]])
        translation = np.array([[t.x], [t.y], [t.z]])
        result = rot_matrix @ point_vec + translation
        return (result[0, 0], result[1, 0], result[2, 0])

    def _transform_bbox_to_map_aabb(self, box_min, box_max, transform: TransformStamped):
        if box_min is None or box_max is None:
            return None, None

        corners_cam = [
            (box_min[0], box_min[1], box_min[2]),
            (box_min[0], box_min[1], box_max[2]),
            (box_min[0], box_max[1], box_min[2]),
            (box_min[0], box_max[1], box_max[2]),
            (box_max[0], box_min[1], box_min[2]),
            (box_max[0], box_min[1], box_max[2]),
            (box_max[0], box_max[1], box_min[2]),
            (box_max[0], box_max[1], box_max[2]),
        ]

        corners_map = []
        for x, y, z in corners_cam:
            p = Vector3(x=float(x), y=float(y), z=float(z))
            corners_map.append(self.transform_point(p, transform))

        corners_np = np.asarray(corners_map, dtype=np.float64)
        min_map = np.min(corners_np, axis=0)
        max_map = np.max(corners_np, axis=0)

        center_map = (
            float((min_map[0] + max_map[0]) * 0.5),
            float((min_map[1] + max_map[1]) * 0.5),
            float((min_map[2] + max_map[2]) * 0.5),
        )
        size_map = (
            max(float(max_map[0] - min_map[0]), 0.01),
            max(float(max_map[1] - min_map[1]), 0.01),
            max(float(max_map[2] - min_map[2]), 0.01),
        )
        return center_map, size_map

    def euclidean_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _dynamic_gate(self, size_a: Tuple[float, float, float], size_b: Tuple[float, float, float]) -> float:
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
        return Vector3(
            x=float(pose_in_camera.x),
            y=float(pose_in_camera.y),
            z=float(pose_in_camera.z),
        )

    def _to_embedding(self, embeddings) -> Optional[np.ndarray]:
        if embeddings is None:
            return None
        arr = np.asarray(embeddings, dtype=np.float32)
        return self._normalize_embedding(arr)

    def _stamp_to_ns(self, stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _ema_tuple(self, old, new, alpha: float):
        return tuple((1.0 - alpha) * old[i] + alpha * new[i] for i in range(3))

    def _aabb_iou(self, center_a, size_a, center_b, size_b) -> float:
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
        half = np.array(box_size, dtype=np.float64) / 2.0 + float(max(margin, 0.0))
        p = np.array(point, dtype=np.float64)
        c = np.array(box_center, dtype=np.float64)
        delta = np.abs(p - c)
        return bool(np.all(delta <= half))

    def _choose_consensus_class(
        self,
        class_counts: Dict[str, int],
        class_conf_sums: Dict[str, float],
        current_name: str,
    ) -> str:
        """
        Choose label from both frequency and confidence, with hysteresis.

        Score(class) = count_weight * count + confidence_weight * avg_confidence
        """
        if not class_counts:
            return current_name

        def class_score(name: str) -> float:
            # Weighted combination of frequency and mean confidence per class.
            count = float(class_counts.get(name, 0))
            conf_sum = float(class_conf_sums.get(name, 0.0))
            avg_conf = conf_sum / max(count, 1.0)
            return self.class_count_weight * count + self.class_confidence_weight * avg_conf

        best_name = max(class_counts.keys(), key=class_score)
        if best_name == current_name:
            return best_name

        current_score = class_score(current_name)
        best_score = class_score(best_name)
        current_count = class_counts.get(current_name, 0)

        # Keep current label until we have enough evidence to switch.
        if current_count >= self.min_class_votes_to_lock and best_score < (current_score + self.class_switch_margin):
            return current_name
        return best_name
