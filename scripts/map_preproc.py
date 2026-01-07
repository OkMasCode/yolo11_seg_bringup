import json
import numpy as np
from sklearn.cluster import DBSCAN

MAP_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/map.json"
OUTPUT_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/clustered_map.json"

def load_house_map(file_path):
    print(f"[DEBUG] Loading map from: {file_path}")
    map_objects = {}
    with open(file_path, 'r') as f:
        house_map = json.load(f)
    for obj_id, obj in house_map.items():
        name = obj['name']
        x = obj['pose_map']['x']
        y = obj['pose_map']['y']
        z = obj['pose_map']['z']
        map_objects[obj_id] = {
            'name': name,
            'coords': (x, y, z)
        }
    print(f"[DEBUG] Loaded {len(map_objects)} objects from map")
    return map_objects

def cluster_map(object_dict, eps=1.5, min_samples=2):
    """
    Generates a map where each object has its ID, cluster assignment, class name, and coordinates.
    
    Args:
        object_dict (dict): {'object_id': {'name': str, 'coords': (x, y, z)}}
    """
    print(f"[DEBUG] Starting clustering with eps={eps}, min_samples={min_samples}")
    # 1. Handle empty input
    if not object_dict:
        print("[DEBUG] Empty object dictionary!")
        return []

    obj_ids = list(object_dict.keys())
    coords = [obj['coords'] for obj in object_dict.values()]
    names = [obj['name'] for obj in object_dict.values()]
    X = np.array(coords)
    print(f"[DEBUG] Prepared {len(obj_ids)} objects for clustering")

    # 2. Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(X)
    labels = clustering.labels_
    print(f"[DEBUG] DBSCAN completed. Labels: {labels}")

    # 2.5. Reassign outliers to their own unique cluster IDs
    outlier_indices = np.where(labels == -1)[0]
    if len(outlier_indices) > 0:
        max_cluster_id = max(labels[labels != -1]) if len(labels[labels != -1]) > 0 else -1
        next_cluster_id = max_cluster_id + 1
        for i, outlier_idx in enumerate(outlier_indices):
            labels[outlier_idx] = next_cluster_id + i
            print(f"[DEBUG] Reassigned outlier at index {outlier_idx} to cluster {next_cluster_id + i}")

    # 3. Compute centroids for each cluster (including single-object clusters)
    unique_labels = sorted(set(labels))
    centroid_by_label = {}
    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0]
        centroid = X[indices].mean(axis=0)
        centroid_by_label[int(lbl)] = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
        print(f"[DEBUG] Centroid for cluster {lbl}: {centroid_by_label[int(lbl)]}")

    # 4. Create output list with all objects
    final_output = []

    # 5. Create object entries with id, cluster, class, coords, and cluster centroid
    for obj_id, name, coord, label in zip(obj_ids, names, coords, labels):
        # Determine centroid for the object's cluster
        cx, cy, cz = centroid_by_label[int(label)]
        centroid_obj = {"x": cx, "y": cy, "z": cz}

        obj_data = {
            "id": obj_id,
            "cluster": int(label),
            "class": name,
            "coords": {
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2])
            },
            "cluster_centroid": centroid_obj
        }
        
        print(f"[DEBUG] Object '{obj_id}' (class: {name}) assigned to cluster {label}")
        
        final_output.append(obj_data)

    print(f"[DEBUG] Clustering complete. Final output has {len(final_output)} objects")
    return final_output

def write_map_to_file(clustered_map, file_path):
    print(f"[DEBUG] Writing {len(clustered_map)} entries to: {file_path}")
    with open(file_path, 'w') as f:
        json.dump(clustered_map, f, indent=4)
    print(f"[DEBUG] Successfully wrote clustered map to file")

def main():
    print("[DEBUG] Starting map preprocessing...")
    clean_map = load_house_map(MAP_FILE)
    clustered_map = cluster_map(clean_map)
    write_map_to_file(clustered_map, OUTPUT_FILE)
    print("[DEBUG] Map preprocessing complete!")

if __name__ == "__main__":
    main()

