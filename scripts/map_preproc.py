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
    for obj in house_map:
        name = obj['class']
        x = obj['coords']['x']
        y = obj['coords']['y']
        z = obj['coords']['z']
        map_objects[name] = (x, y, z)
    print(f"[DEBUG] Loaded {len(map_objects)} objects from map")
    return map_objects

def cluster_map(object_dict, eps=1.5, min_samples=2):
    """
    Generates a map where clustered objects are grouped, and outliers 
    are listed as standalone entries.
    
    Args:
        object_dict (dict): {'object_name': (x, y, z)}
    """
    print(f"[DEBUG] Starting clustering with eps={eps}, min_samples={min_samples}")
    # 1. Handle empty input
    if not object_dict:
        print("[DEBUG] Empty object dictionary!")
        return []

    names = list(object_dict.keys())
    coords = list(object_dict.values())
    X = np.array(coords)
    print(f"[DEBUG] Prepared {len(names)} objects for clustering")

    # 2. Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(X)
    labels = clustering.labels_
    print(f"[DEBUG] DBSCAN completed. Labels: {labels}")

    # 3. Prepare containers
    # clusters_map: { cluster_id: [list_of_object_dicts] }
    clusters_map = {}
    # final_output: The list that will be dumped to JSON
    final_output = []

    # 4. Sort items into Clusters or Outliers
    for name, coord, label in zip(names, coords, labels):
        
        # Create the standard object dictionary
        obj_data = {
            "class": name,
            "coords": {
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2])
            }
        }

        if label == -1:
            # CASE A: Outlier (Noise)
            # Add directly to the main list
            print(f"[DEBUG] Object '{name}' is an outlier")
            final_output.append(obj_data)
        else:
            # CASE B: Clustered
            # Add to the temporary cluster grouping
            c_id = int(label)
            if c_id not in clusters_map:
                clusters_map[c_id] = []
            clusters_map[c_id].append(obj_data)

    # 5. Format the clusters and add them to final output
    for c_id, items in clusters_map.items():
        print(f"[DEBUG] Cluster {c_id} has {len(items)} items")
        cluster_entry = {
            "cluster_id": c_id,
            "items": items  # The list of objects inside this cluster
        }
        final_output.append(cluster_entry)

    print(f"[DEBUG] Clustering complete. Final output has {len(final_output)} entries")
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

