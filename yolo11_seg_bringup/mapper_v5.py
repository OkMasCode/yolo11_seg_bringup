import json
import math
import os
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
import open3d as o3d
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
    accumulated_points: np.ndarray  # <--- NEW: Stores the geometry
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

    @property
    def obb(self) -> o3d.geometry.OrientedBoundingBox:
        # 1. Create the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.accumulated_points)
        
        # 2. SOTA Fix: Statistical Outlier Removal
        # This removes the invisible "dust" that causes the bounding box to explode.
        # It looks at the 20 nearest neighbors for every point. If a point is further 
        # away than 2.0 standard deviations from the average distance, it is deleted.
        if len(pcd.points) > 20:
            clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Fallback: If SOR accidentally deleted the whole object, use the original points
            if len(clean_pcd.points) > 5:
                return clean_pcd.get_minimal_oriented_bounding_box()
                
        # 3. Fit the tight, rotated bounding box around the clean geometry
        return pcd.get_minimal_oriented_bounding_box()

    @property
    def box_size(self) -> Tuple[float, float, float]:
        # Return the extents (length, width, height) of the oriented box
        return tuple(float(x) for x in self.obb.extent)

    @property
    def pose_map(self) -> Tuple[float, float, float]:
        centroid = np.mean(self.accumulated_points, axis=0)
        return tuple(float(x) for x in centroid)
        
    @property
    def pose_cam(self) -> Tuple[float, float, float]:
        # pose_cam is deprecated in this architecture but kept to prevent downstream errors
        return self.pose_map 

@dataclass
class TentativeTrack:
    track_id: str
    frame: str
    timestamp: Time
    accumulated_points: np.ndarray # <--- NEW: Stores the geometry
    hits: int
    first_seen_ns: int
    last_seen_ns: int
    class_name: str
    confidence_max: float = 0.0
    confidence_sum: float = 0.0
    image_embedding: Optional[np.ndarray] = None
    embedding_confidence_max: float = -1.0
    
    @property
    def obb(self) -> o3d.geometry.OrientedBoundingBox:
        # 1. Create the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.accumulated_points)
        
        # 2. SOTA Fix: Statistical Outlier Removal
        # This removes the invisible "dust" that causes the bounding box to explode.
        # It looks at the 20 nearest neighbors for every point. If a point is further 
        # away than 2.0 standard deviations from the average distance, it is deleted.
        if len(pcd.points) > 20:
            clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Fallback: If SOR accidentally deleted the whole object, use the original points
            if len(clean_pcd.points) > 5:
                return clean_pcd.get_minimal_oriented_bounding_box()
                
        # 3. Fit the tight, rotated bounding box around the clean geometry
        return pcd.get_minimal_oriented_bounding_box()

    @property
    def box_size(self) -> Tuple[float, float, float]:
        # Return the extents (length, width, height) of the oriented box
        return tuple(float(x) for x in self.obb.extent)

    @property
    def pose_map(self) -> Tuple[float, float, float]:
        centroid = np.mean(self.accumulated_points, axis=0)
        return tuple(float(x) for x in centroid)
        
    @property
    def pose_cam(self) -> Tuple[float, float, float]:
        # pose_cam is deprecated in this architecture but kept to prevent downstream errors
        return self.pose_map 


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



class SemanticObjectMapV5:
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

        # Periodic clustering cleanup (keep only largest connected cluster per object).
        self.enable_largest_cluster_filter = False
        self.cluster_filter_eps_m = 0.18
        self.cluster_filter_min_points = 12

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
        self.confidence_ema_alpha = 0.20

        # Class consensus (count + confidence) with hysteresis against rapid flips.
        self.class_count_weight = 1.0
        self.class_confidence_weight = 2.0
        self.class_switch_margin = 0.75
        self.min_class_votes_to_lock = 4

        self.node.get_logger().info(
            "[mapper_v5:init] "
            f"min_input_confidence={self.min_input_confidence} "
            f"depth_range=({self.min_detection_depth_m},{self.max_detection_depth_m}) "
            f"iou_min={self.min_association_iou} "
            f"confirm_hits={self.confirmation_min_hits} "
            f"confirm_min_age={self.confirmation_min_age_sec}s"
        )

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

    def add_detections_batch(
        self,
        object_names: list[str],
        tracker_ids: list[str],
        detection_stamp,
        camera_frame: str = 'camera_color_optical_frame',
        fixed_frame: str = 'map',
        embeddings_list: list = None,
        confidences: list[float] = None,
        points_cam_list: list[np.ndarray] = None,
    ) -> None:
        """
        Processes a full frame of detections. 
        Trusts the tracker ID first, then uses Bipartite Matching for new IDs.
        """
        if not points_cam_list or len(points_cam_list) == 0:
            return

        try:
            lookup_time = rclpy.time.Time.from_msg(detection_stamp)
            transform = self.tf_buffer.lookup_transform(
                fixed_frame, camera_frame, lookup_time, timeout=Duration(seconds=0.8)
            )
        except TransformException as ex:
            self.node.get_logger().warning(f"[mapper_v5] TF Error: {ex}")
            return

        current_ns = self._stamp_to_ns(detection_stamp)
        self._prune_stale_state(current_ns)

        # ---------------------------------------------------------
        # 1. PREPARE VALID DETECTIONS
        # ---------------------------------------------------------
        valid_detections = []
        for i in range(len(points_cam_list)):
            pts_cam = points_cam_list[i]
            if pts_cam is None or len(pts_cam) < 5:
                continue

            # Quality Gate
            conf = confidences[i] if confidences else 0.0
            depth_m = float(np.mean(pts_cam[:, 2]))
            
            pts_map = self.transform_pointcloud(pts_cam, transform)
            obs_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_map))
            obs_obb = obs_pcd.get_minimal_oriented_bounding_box()
            obs_size = obs_obb.extent

            gate_failure = self._quality_gate_failure_reason(conf, depth_m, obs_size)
            if gate_failure:
                continue

            emb = self._to_embedding(embeddings_list[i] if embeddings_list else None)

            valid_detections.append({
                'name': object_names[i],
                'track_id': tracker_ids[i],
                'points_map': pts_map,
                'obb': obs_obb,
                'centroid': np.mean(pts_map, axis=0),
                'embedding': emb,
                'confidence': conf
            })

        if not valid_detections:
            return

        unmatched_detections = []

        # ---------------------------------------------------------
        # 2. PHASE 1: DIRECT ID ROUTING (TRUST THE TRACKER)
        # ---------------------------------------------------------
        for det in valid_detections:
            t_id = det['track_id']
            matched = False

            # Check if bound to a confirmed Map Object
            bound_map_id = self.track_to_map.get(t_id)
            if bound_map_id in self.objects:
                map_obj = self.objects[bound_map_id]
                
                # SOTA SAFETY GATE: Do not trust the tracker if the YOLO class suddenly flips
                if det['name'] != map_obj.current_name:
                    self.node.get_logger().warning(
                        f"[mapper_v5] Tracker {t_id} flipped class: {map_obj.current_name} -> {det['name']}. "
                        "Revoking trust, sending to matrix."
                    )
                    # We do NOT set matched = True, so it falls to the unmatched list
                else:
                    self._update_object(
                        map_id=bound_map_id, object_name=det['name'], detection_stamp=detection_stamp,
                        points_map=det['points_map'], similarity=0.0, confidence=det['confidence'],
                        image_embedding=det['embedding'], current_ns=current_ns, source_track_id=t_id
                    )
                    self.track_last_seen_ns[t_id] = current_ns
                    matched = True
            
            # Check if it is an active Tentative Track
            elif t_id in self.tentative_tracks:
                track = self.tentative_tracks[t_id]
                
                # Apply the same safety gate here
                if det['name'] != track.class_name:
                    self.node.get_logger().warning(
                        f"[mapper_v5] Tentative Tracker {t_id} flipped class. Revoking trust."
                    )
                else:
                    self._update_tentative(
                        object_name=det['name'], tracker_id=t_id, points_map=det['points_map'],
                        detection_stamp=detection_stamp, confidence=det['confidence'],
                        image_embedding=det['embedding'], current_ns=current_ns, frame=fixed_frame
                    )
                    matched = True

            if not matched:
                unmatched_detections.append(det)

        if not unmatched_detections:
            # Everything was handled by the tracker IDs
            return

        # ---------------------------------------------------------
        # 3. PHASE 2: BIPARTITE MATCHING FOR NEW IDs
        # ---------------------------------------------------------
        map_ids = list(self.objects.keys())
        
        # If the map is empty, all unmatched detections become new tentative tracks
        if not map_ids:
            for det in unmatched_detections:
                self._update_tentative(
                    object_name=det['name'], tracker_id=det['track_id'], points_map=det['points_map'],
                    detection_stamp=detection_stamp, confidence=det['confidence'],
                    image_embedding=det['embedding'], current_ns=current_ns, frame=fixed_frame
                )
            return

        N = len(unmatched_detections)
        M = len(map_ids)
        cost_matrix = np.zeros((N, M), dtype=np.float32)

        # ---------------------------------------------------------
        # 3. PHASE 2: BIPARTITE MATCHING FOR NEW IDs
        # ---------------------------------------------------------
        # ... (setup code remains the same) ...
        
        # Tuning Weights 
        w_dist = 1.0
        w_iou = 1.0
        w_sem = 2.5 
        MAX_COST = 3.5 

        for i, det in enumerate(unmatched_detections):
            for j, m_id in enumerate(map_ids):
                map_obj = self.objects[m_id]

                # A. Spatial Cost
                dist = float(np.linalg.norm(det['centroid'] - map_obj.pose_map))
                if dist > 1.5: 
                    cost_matrix[i, j] = 999.0
                    continue

                # B. Geometric Cost (1.0 - IoU)
                iou = self.compute_obb_iou(det['points_map'], det['obb'], map_obj.accumulated_points, map_obj.obb)
                cost_iou = 1.0 - iou

                # C. Semantic Cost
                cost_sem = self.compute_semantic_distance(det['embedding'], map_obj.image_embedding)

                # D. Explicit Class Mismatch Penalty
                class_penalty = 0.0
                if det['name'] != map_obj.current_name:
                    # Adding 5.0 pushes the total cost well over the MAX_COST of 3.5,
                    # mathematically forcing the algorithm to reject this match.
                    class_penalty = 5.0 

                # Total Multi-Modal Cost
                cost_matrix[i, j] = (w_dist * dist) + (w_iou * cost_iou) + (w_sem * cost_sem) + class_penalty


        # Solve the optimal assignment
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_detections = set()

        for idx in range(len(row_ind)):
            det_idx = row_ind[idx]
            map_idx = col_ind[idx]
            cost = cost_matrix[det_idx, map_idx]

            if cost < MAX_COST:
                det = unmatched_detections[det_idx]
                m_id = map_ids[map_idx]
                
                self._update_object(
                    map_id=m_id, object_name=det['name'], detection_stamp=detection_stamp,
                    points_map=det['points_map'], similarity=0.0, confidence=det['confidence'],
                    image_embedding=det['embedding'], current_ns=current_ns, source_track_id=det['track_id']
                )
                self.track_to_map[det['track_id']] = m_id
                self.track_last_seen_ns[det['track_id']] = current_ns
                assigned_detections.add(det_idx)

        # ---------------------------------------------------------
        # 4. PHASE 3: SPAWN NEW TRACKS
        # ---------------------------------------------------------
        for i, det in enumerate(unmatched_detections):
            if i not in assigned_detections:
                self._update_tentative(
                    object_name=det['name'], tracker_id=det['track_id'], points_map=det['points_map'],
                    detection_stamp=detection_stamp, confidence=det['confidence'],
                    image_embedding=det['embedding'], current_ns=current_ns, frame=fixed_frame
                )

    def _update_tentative(
        self,
        object_name: str,
        tracker_id: str,
        points_map: np.ndarray, # <--- NEW: Accept the point cloud
        detection_stamp,
        confidence: float,
        image_embedding,
        current_ns: int,
        frame: str,
    ) -> bool:
        track = self.tentative_tracks.get(tracker_id)

        if track is None:
            # Start a new tentative track
            self.tentative_tracks[tracker_id] = TentativeTrack(
                track_id=tracker_id,
                frame=frame,
                timestamp=detection_stamp,
                accumulated_points=points_map, # <--- Store the geometry
                hits=1,
                first_seen_ns=current_ns,
                last_seen_ns=current_ns,
                class_name=object_name,
                confidence_max=float(confidence),
                confidence_sum=max(float(confidence), 0.0),
                image_embedding=self._normalize_embedding(image_embedding),
                embedding_confidence_max=(float(confidence) if image_embedding is not None else -1.0),
            )
            self.node.get_logger().info(
                "[mapper_v5:tentative:new] "
                f"track={tracker_id} class={object_name} hits=1 conf={float(confidence):.3f}"
            )
            return False

        if object_name != track.class_name:
            # Reset track if class flips while still tentative
            self.tentative_tracks[tracker_id] = TentativeTrack(
                track_id=tracker_id,
                frame=frame,
                timestamp=detection_stamp,
                accumulated_points=points_map, # <--- Store the geometry
                hits=1,
                first_seen_ns=current_ns,
                last_seen_ns=current_ns,
                class_name=object_name,
                confidence_max=float(confidence),
                confidence_sum=max(float(confidence), 0.0),
                image_embedding=self._normalize_embedding(image_embedding),
                embedding_confidence_max=(float(confidence) if image_embedding is not None else -1.0),
            )
            self.node.get_logger().warning(
                "[mapper_v5:tentative:class_reset] "
                f"track={tracker_id} old_class={track.class_name} new_class={object_name}"
            )
            return False

        # ==========================================
        # FUSE GEOMETRY (Replaces EMA Math)
        # ==========================================
        updated_points = self._fuse_geometry(track.accumulated_points, points_map)
        hits = track.hits + 1
        conf_sum = track.confidence_sum + max(float(confidence), 0.0)

        # Fuse embeddings normally
        fused_embedding = self._fuse_embeddings_running_avg(
            current_embedding=track.image_embedding,
            current_count=track.hits,
            new_embedding=image_embedding,
            new_count=1,
        )

        best_embedding_conf = float(track.embedding_confidence_max)
        if image_embedding is not None:
            best_embedding_conf = max(best_embedding_conf, float(confidence))

        # Save updated tentative track
        self.tentative_tracks[tracker_id] = TentativeTrack(
            track_id=tracker_id,
            frame=frame,
            timestamp=detection_stamp,
            accumulated_points=updated_points, # <--- The new geometry
            hits=hits,
            first_seen_ns=track.first_seen_ns,
            last_seen_ns=current_ns,
            class_name=track.class_name,
            confidence_max=max(track.confidence_max, float(confidence)),
            confidence_sum=conf_sum,
            image_embedding=fused_embedding,
            embedding_confidence_max=best_embedding_conf,
        )

        # Check for map promotion
        age_ns = current_ns - track.first_seen_ns
        window_ns = int(self.confirmation_time_window_sec * 1e9)
        min_age_ns = int(self.confirmation_min_age_sec * 1e9)
        avg_conf = conf_sum / max(hits, 1)

        promote_ok = (
            hits >= self.confirmation_min_hits
            and age_ns >= min_age_ns
            and age_ns <= window_ns
            and max(track.confidence_max, float(confidence)) >= self.min_confidence_for_promotion
            and avg_conf >= self.min_avg_confidence_for_promotion
        )

        self.node.get_logger().info(
            "[mapper_v5:tentative:update] "
            f"track={tracker_id} class={track.class_name} hits={hits} age_s={age_ns / 1e9:.2f} "
            f"avg_conf={avg_conf:.3f} promote_ok={promote_ok}"
        )

        if not promote_ok:
            return False

        # Confirmation passed: promote to MapObject
        promoted = self.tentative_tracks.pop(tracker_id)
        map_id = self._new_map_id()
        self.objects[map_id] = MapObject(
            map_id=map_id,
            frame=promoted.frame,
            timestamp=promoted.timestamp,
            accumulated_points=promoted.accumulated_points, # <--- Pass the geometry
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
        self.node.get_logger().info(
            "[mapper_v5:tentative:promoted] "
            f"track={tracker_id} map_id={map_id} class={promoted.class_name} hits={promoted.hits}"
        )
        return True

    def _update_object(
        self,
        map_id: str,
        object_name: str,
        detection_stamp,
        points_map: np.ndarray, # <--- NEW
        similarity: float,
        confidence: float,
        image_embedding,
        current_ns: int,
        source_track_id: str,
    ) -> None:
        entry = self.objects[map_id]

        # 1. Fuse the geometry!
        updated_points = self._fuse_geometry(entry.accumulated_points, points_map)

        # 2. Accumulate votes (Keep your existing vote logic here)
        class_votes = dict(entry.class_votes)
        class_votes[object_name] = class_votes.get(object_name, 0.0) + max(float(confidence), 0.01)
        class_counts = dict(entry.class_counts)
        class_counts[object_name] = class_counts.get(object_name, 0) + 1
        class_conf_sums = dict(entry.class_conf_sums)
        class_conf_sums[object_name] = class_conf_sums.get(object_name, 0.0) + max(float(confidence), 0.01)

        best_class = self._choose_consensus_class(class_counts, class_conf_sums, entry.current_name)
        conf_ema = (1.0 - self.confidence_ema_alpha) * entry.confidence_ema + self.confidence_ema_alpha * float(confidence)

        embedding = self._fuse_embeddings_running_avg(
            entry.image_embedding, entry.occurrences, image_embedding, 1
        )
        embedding_confidence_max = max(float(entry.embedding_confidence_max), float(confidence)) if image_embedding is not None else float(entry.embedding_confidence_max)

        # 3. Save the object (pose_map and box_size compute automatically)
        self.objects[map_id] = MapObject(
            map_id=entry.map_id,
            frame=entry.frame,
            timestamp=detection_stamp,
            accumulated_points=updated_points, # <--- The new geometry
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
        self.node.get_logger().info(
            "[mapper_v5:object:update] "
            f"map_id={map_id} prev_name={entry.current_name} new_name={best_class} "
            f"occurrences={entry.occurrences + 1} source_track={source_track_id}"
        )

    def _fuse_objects(self, keep_id: str, drop_id: str) -> None:
        keep = self.objects[keep_id]
        drop = self.objects[drop_id]

        total_hits = keep.occurrences + drop.occurrences

        # ==========================================
        # FUSE GEOMETRY (Replaces Weighted Averages)
        # ==========================================
        updated_points = self._fuse_geometry(keep.accumulated_points, drop.accumulated_points)

        # Merge class votes normally
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

        # Save fused object
        self.objects[keep_id] = MapObject(
            map_id=keep.map_id,
            frame=keep.frame,
            timestamp=keep.timestamp if keep.last_seen_ns >= drop.last_seen_ns else drop.timestamp,
            accumulated_points=updated_points, # <--- The new merged geometry
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
        
        # Delete the consumed object
        del self.objects[drop_id]

        # Reroute any active tracker IDs pointing to the dropped object
        for track_id, mapped_id in list(self.track_to_map.items()):
            if mapped_id == drop_id:
                self.track_to_map[track_id] = keep_id

        self.node.get_logger().warning(
            "[mapper_v5:fuse_objects] "
            f"keep_id={keep_id} drop_id={drop_id} total_occurrences={total_hits}"
        )

    def _prune_stale_state(self, current_ns: int) -> None:
        stale_tentative_ns = int(self.tentative_max_stale_sec * 1e9)
        stale_binding_ns = int(self.binding_ttl_sec * 1e9)
        removed_tentative = 0
        removed_bindings = 0

        # Drop tentative tracks that stopped receiving observations.
        for track_id, t in list(self.tentative_tracks.items()):
            if current_ns - t.last_seen_ns > stale_tentative_ns:
                del self.tentative_tracks[track_id]
                removed_tentative += 1

        # Drop old tracker bindings to avoid reconnecting stale IDs to map objects.
        for track_id, last_seen in list(self.track_last_seen_ns.items()):
            if current_ns - last_seen > stale_binding_ns:
                self.track_last_seen_ns.pop(track_id, None)
                self.track_to_map.pop(track_id, None)
                removed_bindings += 1

        if removed_tentative > 0 or removed_bindings > 0:
            self.node.get_logger().info(
                "[mapper_v5:prune] "
                f"removed_tentative={removed_tentative} removed_bindings={removed_bindings}"
            )

    def _new_map_id(self) -> str:
        map_id = f"map_obj_{self._next_map_id:06d}"
        self._next_map_id += 1
        return map_id

    def export_to_json(self, directory_path: str, file: str = 'map_v5.json') -> None:
        os.makedirs(directory_path, exist_ok=True)
        path = os.path.join(directory_path, file)
        self.node.get_logger().info(
            f"[mapper_v5:export:start] path={path} objects={len(self.objects)}"
        )
        
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
            corners = np.asarray(obj.obb.get_box_points())
            corners_list = [{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for p in corners]
            export_data[map_id] = {
                'name': obj.current_name,
                'frame': obj.frame,
                'timestamp': {'sec': obj.timestamp.sec, 'nanosec': obj.timestamp.nanosec},
                'pose_map': {'x': float(obj.pose_map[0]), 'y': float(obj.pose_map[1]), 'z': float(obj.pose_map[2])},
                'bbox_type': 'obb',
                'box_size': {'x': float(obj.box_size[0]), 'y': float(obj.box_size[1]), 'z': float(obj.box_size[2])},
                'bbox_corners': corners_list,
                'occurrences': int(obj.occurrences),
                'similarity': float(obj.similarity),
                'image_embedding': obj.image_embedding.tolist() if obj.image_embedding is not None else None,
                'embedding_confidence': float(obj.embedding_confidence_max),
                'confidence': float(obj.confidence_ema),
            }

        with open(path, 'w') as json_file:
            json.dump(export_data, json_file, indent=4)
        self.node.get_logger().info(
            f"[mapper_v5:export:done] path={path} objects={len(export_data)}"
        )

    def load_from_json(self, directory_path: str, file: str = 'map_v5.json') -> None:
        path = os.path.join(directory_path, file)
        self.node.get_logger().info(f"[mapper_v5:load:start] path={path}")
        if not os.path.exists(path):
            self.node.get_logger().info(f"[mapper_v5] No map file found at {path}, starting empty")
            return

        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
            if not isinstance(data, dict):
                self.node.get_logger().warning(
                    f"[mapper_v5:load:reject] root is not dict type={type(data)}"
                )
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
            self.node.get_logger().info(f"[mapper_v5] Loaded {len(self.objects)} objects from {path}")

        except Exception as ex:
            self.node.get_logger().error(f"[mapper_v5] Error loading map from {path}: {ex}")
            self.node.get_logger().error(traceback.format_exc())

    def transform_point(self, point, transform: TransformStamped) -> Tuple[float, float, float]:
        q = transform.transform.rotation
        t = transform.transform.translation
        rot_matrix = quaternion_matrix([q.w, q.x, q.y, q.z])[:3, :3]
        point_vec = np.array([[point.x], [point.y], [point.z]])
        translation = np.array([[t.x], [t.y], [t.z]])
        result = rot_matrix @ point_vec + translation
        return (result[0, 0], result[1, 0], result[2, 0])

    def _passes_detection_quality_gate(
        self,
        confidence: float,
        depth_m: float,
        box_size: Tuple[float, float, float],
    ) -> bool:
        return self._quality_gate_failure_reason(confidence, depth_m, box_size) is None

    def _quality_gate_failure_reason(
        self,
        confidence: float,
        depth_m: float,
        box_size: Tuple[float, float, float],
    ) -> Optional[str]:
        if float(confidence) < self.min_input_confidence:
            return f"confidence_low({float(confidence):.3f}<{self.min_input_confidence:.3f})"
        if not math.isfinite(float(depth_m)):
            return "depth_not_finite"
        if float(depth_m) < self.min_detection_depth_m or float(depth_m) > self.max_detection_depth_m:
            return (
                f"depth_out_of_range({float(depth_m):.3f} not in "
                f"[{self.min_detection_depth_m:.3f},{self.max_detection_depth_m:.3f}])"
            )

        extent = max(float(box_size[0]), float(box_size[1]), float(box_size[2]))
        if extent < self.min_box_extent_m or extent > self.max_box_extent_m:
            return (
                f"extent_out_of_range({extent:.3f} not in "
                f"[{self.min_box_extent_m:.3f},{self.max_box_extent_m:.3f}])"
            )

        volume = float(box_size[0] * box_size[1] * box_size[2])
        if volume < self.min_box_volume_m3 or volume > self.max_box_volume_m3:
            return (
                f"volume_out_of_range({volume:.6f} not in "
                f"[{self.min_box_volume_m3:.6f},{self.max_box_volume_m3:.6f}])"
            )

        return None

    def _to_embedding(self, embeddings) -> Optional[np.ndarray]:
        if embeddings is None:
            return None
        arr = np.asarray(embeddings, dtype=np.float32)
        return self._normalize_embedding(arr)

    def _stamp_to_ns(self, stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

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

    def transform_pointcloud(self, points: np.ndarray, transform: TransformStamped) -> np.ndarray:
        """Vectorized transformation of points from camera to map frame."""
        if points is None or len(points) == 0:
            return np.empty((0, 3))
        q = transform.transform.rotation
        t = transform.transform.translation
        rot_matrix = quaternion_matrix([q.w, q.x, q.y, q.z])[:3, :3]
        translation = np.array([t.x, t.y, t.z])
        transformed_points = (rot_matrix @ points.T).T + translation
        return transformed_points.astype(np.float32)

    def _fuse_geometry(self, old_points: np.ndarray, new_points: np.ndarray) -> np.ndarray:
        """Merges point clouds and voxel downsamples to 5cm resolution."""
        if old_points is None or len(old_points) == 0: return new_points
        if new_points is None or len(new_points) == 0: return old_points

        combined = np.vstack([old_points, new_points])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined)
        
        downsampled = pcd.voxel_down_sample(voxel_size=0.1)
        return np.asarray(downsampled.points, dtype=np.float32)

    def compute_obb_iou(self, points1: np.ndarray, obb1: o3d.geometry.OrientedBoundingBox, 
                              points2: np.ndarray, obb2: o3d.geometry.OrientedBoundingBox) -> float:
        """
        Computes a point-to-OBB overlap proxy for IoU.
        Instead of complex analytical box intersection, we count how many points 
        from cloud 1 fall tightly into box 2, and vice-versa.
        """
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
            
        vec1 = o3d.utility.Vector3dVector(points1)
        vec2 = o3d.utility.Vector3dVector(points2)
        
        # Get indices of points inside the other's bounding box using built-in Open3D
        in_2 = obb2.get_point_indices_within_bounding_box(vec1)
        in_1 = obb1.get_point_indices_within_bounding_box(vec2)
        
        # Average the intersecting points for robustness
        intersect_count = (len(in_2) + len(in_1)) / 2.0
        
        # Union is total points minus the intersection
        union_count = len(points1) + len(points2) - intersect_count
        
        if union_count <= 0: 
            return 0.0
            
        return float(intersect_count / union_count)
    
    def _refine_object_geometry(self, map_id: str) -> None:
        """
        Uses DBSCAN clustering to isolate the largest contiguous physical mass 
        of the object, discarding odometry smears and disconnected noise.
        """
        if map_id not in self.objects:
            return

        obj = self.objects[map_id]
        points = obj.accumulated_points
        original_count = len(points)

        # Require a minimum number of points to perform clustering
        if original_count < 20:
            return 

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # ==========================================
        # DBSCAN CLUSTERING
        # ==========================================
        # eps = 0.18: Must be larger than the voxel downsample size (0.10) to keep the main object connected.
        # min_points = 10: Minimum local density to form a cluster.
        labels = np.array(pcd.cluster_dbscan(eps=0.18, min_points=10, print_progress=False))
        
        # If no clusters were found (all noise), do nothing
        if labels.max() < 0:
            return

        # Find the label of the largest cluster
        # bincount throws an error on negative numbers, so we shift labels by +1
        # Noise is label -1, which becomes index 0. We ignore index 0 when finding the max.
        shifted_labels = labels + 1
        counts = np.bincount(shifted_labels)
        
        # Ensure there is at least one valid cluster (index > 0)
        if len(counts) <= 1:
            return
            
        # Get the label with the highest count (ignoring noise at index 0)
        largest_cluster_label = counts[1:].argmax()
        
        # Get the indices of the points belonging to the largest cluster
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        new_count = len(largest_cluster_indices)

        # ==========================================
        # UPDATE & DEBUG LOGGING
        # ==========================================
        deleted = original_count - new_count
        
        if deleted > 0 and new_count >= 10:
            # Extract only the points of the largest cluster
            refined_points = points[largest_cluster_indices]
            obj.accumulated_points = refined_points
            
            self.node.get_logger().info(
                f"[mapper_v5:refinement] {map_id} DBSCAN: {original_count} -> {new_count} (Dropped {deleted} smear points)"
            )

    def resolve_overlapping_duplicates(self):
        """
        Scans confirmed objects and merges duplicates ONLY if they share the same class, 
        have highly similar embeddings, and physically overlap.
        """
        ids = list(self.objects.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                if id1 not in self.objects or id2 not in self.objects:
                    continue
                
                obj1, obj2 = self.objects[id1], self.objects[id2]
                
                # ==========================================
                # 1. STRICT CLASS GATE
                # ==========================================
                # If they are not the same class, NEVER merge them.
                if obj1.current_name != obj2.current_name:
                    continue
                
                # ==========================================
                # 2. SEMANTIC GATE
                # ==========================================
                # Even if classes match, ensure their AI fingerprints align 
                # to prevent merging two distinct chairs placed next to each other.
                sem_dist = self.compute_semantic_distance(obj1.image_embedding, obj2.image_embedding)
                if sem_dist > 0.40: # Require strong similarity
                    continue

                # ==========================================
                # 3. GEOMETRIC GATE
                # ==========================================
                iou = self.compute_obb_iou(obj1.accumulated_points, obj1.obb, 
                                           obj2.accumulated_points, obj2.obb)
                
                if iou > 0.15:
                    self.node.get_logger().info(
                        f"[mapper_v5] Resolving true duplicate: {obj1.current_name} ({id1} <-> {id2})"
                    )
                    self._fuse_objects(id1, id2)
                    
                    # Only run refinement after a legitimate merge
                    self._refine_object_geometry(id1)

    def compute_semantic_distance(self, emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> float:
        if emb1 is None or emb2 is None:
            return 1.0
        
        e1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        e2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        sim = np.dot(e1, e2)
        sim = max(-1.0, min(1.0, float(sim))) 
        return 1.0 - sim
