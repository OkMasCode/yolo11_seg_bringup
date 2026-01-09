# YOLO11 Segmentation Bringup Package

ROS2 package for real-time object detection and semantic mapping using YOLOv11 segmentation with CLIP embeddings and 3D pointcloud generation.

## Overview

This package provides a complete pipeline for detecting objects in RGB-D camera streams, computing semantic embeddings with CLIP, generating 3D pointclouds, and building a semantic map of the environment.

## Package Structure

```
yolo11_seg_bringup/
├── yolo11_seg_bringup/               # Python package source
│   ├── __init__.py
│   ├── vision_node.py                # RGB-D YOLO+CLIP node with depth processing
│   ├── flat_vision_node.py           # RGB-only node (no publishers, logging only)
│   ├── mapper_node.py                # Semantic mapping node
│   ├── mapper.py                     # Semantic map data structure
│   └── utils/                        # Utility modules
│       ├── __init__.py
│       ├── clip_processor.py         # SigLIP/CLIP embedding processing
│       └── pointcloud_processor.py   # GPU-accelerated pointcloud generation
├── scripts/
│   ├── llm_interpreter.py            # LLM-based navigation instruction interpreter
│   └── map_preproc.py                # Semantic map clustering preprocessor
├── launch/
│   └── yolo_mapper_reader.launch.py  # Main launch file
├── config/
│   ├── map.json                      # Exported semantic map (raw)
│   ├── clustered_map.json            # Spatially clustered semantic map
│   └── robot_command.json            # LLM-generated navigation commands
```
## LLM Helper Script (Non-ROS Script)

This standalone Python script connects to a local Ollama server for semantic reasoning about the environment. It is **not a ROS2 node** but a complementary tool for interactive navigation planning.

### `llm_interpreter.py` - Interactive Navigation Instruction Interpreter

**Purpose:** Real-time LLM-based interpreter that processes natural language navigation instructions and generates a structured JSON command file (`robot_command.json`) for robot navigation.

**What it does:**
Takes a natural language prompt from the user (e.g., "Go to the black truck", "Bring me the red chair") and generates a `robot_command.json` file containing:

1. **Single class goal** - The target object class extracted from the prompt (e.g., "truck", "chair")
2. **Goal objects** - All instances of that class already present in the semantic map (`config/map.json`), with their IDs and 3D coordinates
3. **Cluster prediction** - The most likely cluster (spatial grouping) where the object can be found, including:
   - Cluster ID
   - List of object classes in that cluster
   - LLM-generated reasoning for cluster selection
   - Centroid coordinates of the cluster
4. **CLIP prompts** - Multiple text variations for visual object localization (e.g., "a red chair", "photo of a red chair", "chair with red cushion")
5. **Action type** - Classification of robot behavior: `go_to_object` (navigate and stay) or `bring_back_object` (fetch and return)

**Processing Pipeline:**
1. **Goal extraction** - LLM extracts the target object class from user prompt with synonym mapping (e.g., "fridge" → "refrigerator")
2. **Map lookup** - Queries `config/map.json` to find all existing instances of the goal class
3. **Cluster prediction** - LLM analyzes the semantic map structure to predict which spatial cluster contains the target
4. **CLIP prompt generation** - Creates 3 text variations incorporating visual features from the user prompt
5. **Action classification** - Determines robot behavior (navigate vs. fetch-and-return)
6. **Output generation** - Saves structured command to `config/robot_command.json`

**Key Features:**
- Multi-stage LLM reasoning with household object dictionary (40+ items)
- Cluster-based spatial reasoning for efficient navigation
- Handles missing objects with semantic fallback to similar clusters
- Per-step computation timing for optimization
- Interactive terminal interface with real-time feedback

**Example Input/Output:**

**User Input:**
```
Navigation Instruction: go to the black chair
```

**Generated `robot_command.json`:**
```json
{
    "timestamp": 1767796783.111627,
    "prompt": "go to the black chair",
    "goal": "chair",
    "goal_objects": [
        {
            "id": "chair_inst24_308_492000000",
            "coords": {"x": -5.37, "y": 6.31, "z": 0.60}
        },
        {
            "id": "chair_inst24_309_526000000",
            "coords": {"x": -6.70, "y": 6.42, "z": 1.11}
        }
    ],
    "cluster_info": {
        "cluster_id": 0,
        "objects": ["chair", "chair", "table", "fridge", "cup"],
        "reasoning": "Black chair is likely in cluster 0, since there are other chairs and it looks like a dining room",
        "coords": {"x": -7.12, "y": 6.36, "z": 0.99}
    },
    "clip_prompt": [
        "a black chair",
        "photo of a black chair",
        "chair of color black"
    ],
    "action": "go_to_object",
    "valid": true
}
```

**How to run it:**
```bash
# 1) Start the Ollama container (if not running)
docker start robot_brain
sleep 5

# 2) Run the script
cd ~/ros2_ws/src/yolo11_seg_bringup
python scripts/llm_interpreter.py

# 3) Enter navigation instructions (interactive loop)
Navigation Instruction: Go to the kitchen
Navigation Instruction: Bring me the remote control

# 4) Press Ctrl+C to exit, then stop the container
docker stop robot_brain
```

**Container Setup (Ollama on Jetson)**

This configuration emphasizes stability (uses `-t`), isolation (dedicated port `11435`), and on-demand usage (no auto-restart to save RAM).

**Phase 1: One-time install**
```bash
# Clean any old container
docker rm -f robot_brain

# Persistent storage
mkdir -p ~/my_robot_models
chmod 777 ~/my_robot_models

# Create the container (does not auto-start on boot)
docker run --runtime nvidia -d -t \
    --network host \
    --name robot_brain \
    --restart no \
    -e OLLAMA_HOST=0.0.0.0:11435 \
    -v ~/my_robot_models:/data \
    dustynv/ollama:0.6.8-r36.4-cu126-22.04

# Pull the model (targeting port 11435)
docker exec -it -e OLLAMA_HOST=0.0.0.0:11435 robot_brain ollama pull llama3.2:3b
```

**Phase 2: Python configuration**
Ensure your script points to the custom port:
```python
OLLAMA_HOST = "http://localhost:11435"
client = ollama.Client(host=OLLAMA_HOST)
```

**Phase 3: Daily usage**
```bash
# 1) Start the brain
docker start robot_brain
# wait ~5s

# 2) Run the script
cd ~/ros2_ws/src/yolo11_seg_bringup
python scripts/llm_interpreter.py

# 3) Stop to free RAM
docker stop robot_brain
```

---

### `map_preproc.py` - Semantic Map Clustering Preprocessor

**Purpose:** Offline preprocessing script that spatially clusters objects in the semantic map to enable efficient room-based reasoning for the LLM interpreter.

**What it does:**
- Loads the raw semantic map from `config/map.json`
- Applies DBSCAN clustering algorithm to group spatially nearby objects
- Assigns outlier objects to unique individual clusters (no object left unclustered)
- Computes cluster centroids for navigation planning
- Outputs structured `config/clustered_map.json` with cluster assignments

**Output Format:**
Each object in the clustered map contains:
- `id` - Unique object identifier
- `cluster` - Cluster ID (spatial grouping)
- `class` - Object class name (e.g., "chair", "table")
- `coords` - Object's 3D position {x, y, z}
- `cluster_centroid` - Centroid of the cluster {x, y, z}

**Clustering Parameters:**
- `eps` (default: 1.5m) - Maximum distance between objects in the same cluster
- `min_samples` (default: 2) - Minimum objects to form a dense cluster

**How to run:**
```bash
cd ~/ros2_ws/src/yolo11_seg_bringup
python scripts/map_preproc.py
```

**When to run:**
- After building a new semantic map with the mapper node
- When the environment layout changes significantly
- Before using `llm_interpreter.py` for the first time

**Note:** This is a preprocessing step. The LLM interpreter reads the clustered map but does not require this script to run each time.

---

## Nodes

### 1. YOLO Segmentation Node (`yolo11_seg_node_main.py`)

**Purpose:** Performs real-time object detection and instance segmentation on RGB-D camera streams using YOLOv11, computes CLIP embeddings for each detection, and generates colored 3D pointclouds.

**Subscribed Topics:**
- `/camera/camera/color/image_raw` (sensor_msgs/Image) - RGB image stream
- `/camera/camera/aligned_depth_to_color/image_raw` (sensor_msgs/Image) - Aligned depth image
- `/camera/camera/color/camera_info` (sensor_msgs/CameraInfo) - Camera intrinsic parameters

**Published Topics:**
- `/yolo/detections` (yolo11_seg_interfaces/DetectedObject) - Per-detection messages with centroid, embeddings, and metadata
- `/yolo/pointcloud` (sensor_msgs/PointCloud2) - Colored 3D pointcloud with class and instance IDs
- `/yolo/centroids` (visualization_msgs/MarkerArray) - Centroid markers for visualization
- `/yolo/annotated` (sensor_msgs/Image) - Annotated RGB image (optional)
- `/yolo/clip_boxes` (sensor_msgs/Image) - CLIP crop visualization (optional)

**Key Features:**
- YOLOv11 instance segmentation with tracking
- CLIP embedding generation for semantic similarity
- GPU-accelerated pointcloud generation

---



### 2. Semantic Mapper Node (`mapper_node2.py`)

**Purpose:** Builds and maintains a semantic map by aggregating detections over time, merging nearby detections into single objects, and transforming coordinates to a fixed reference frame.

**Subscribed Topics:**
- `/yolo/detections` (yolo11_seg_interfaces/DetectedObject) - Incoming detections from YOLO node

**Published Topics:**
- `/yolo/semantic_map` (yolo11_seg_interfaces/SemanticObjectArray) - Complete semantic map snapshot (all stored objects)

**Exported Files:**
- `detections.csv` - Periodic export of semantic map (every 5 seconds)
- `detections_final.csv` - Final export on node shutdown
- `config/map.json` - JSON format semantic map

**Key Features:**
- Spatial deduplication (merges detections within 20cm threshold)
- TF2-based coordinate transformation to fixed frame
- Occurrence counting for detection confidence
- CLIP similarity scoring
- Automatic periodic export

---

### 3. Semantic Map Reader (`clip_reader.py`)

**Purpose:** Subscribes to the semantic map topic and prints all detected objects with their names, 3D coordinates, and similarity scores for debugging and monitoring.

**Subscribed Topics:**
- `/yolo/semantic_map` (yolo11_seg_interfaces/SemanticObjectArray) - Semantic map updates

**Output:**
Logs each object in the format:
```
name=<class_name>, coords=(<x>, <y>, <z>), similarity=<score>
```

**Key Features:**
- Real-time semantic map monitoring
- Human-readable output format
- Useful for debugging and verification

---

## Installation

### Prerequisites
- ROS2 (Humble or later)
- Python 3.8+
- CUDA-capable GPU (recommended)
- YOLOv11 model file (`.engine` or `.pt`)

### Dependencies
```bash
# Install Python dependencies
pip install ultralytics torch torchvision clip opencv-python pillow numpy

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select yolo11_seg_interfaces yolo11_seg_bringup
source install/setup.bash
```

## Usage

### Option 1: Launch All Nodes Together (Recommended)

```bash
# Source your workspace
source ~/ros2_ws/install/setup.bash

# Launch with default parameters
ros2 launch yolo11_seg_bringup yolo_mapper_reader.launch.py

# Launch with custom parameters
ros2 launch yolo11_seg_bringup yolo_mapper_reader.launch.py \
    model_path:=/path/to/your/model.engine \
    text_prompt:="a photo of a bottle" \
    map_frame:=map \
    camera_frame:=camera_link
```

**Available Launch Arguments:**
- `model_path` - Path to YOLO model file (default: `/home/sensor/yolov8n-seg.engine`)
- `image_topic` - RGB image topic (default: `/camera/camera/color/image_raw`)
- `depth_topic` - Depth image topic (default: `/camera/camera/aligned_depth_to_color/image_raw`)
- `camera_info_topic` - Camera info topic (default: `/camera/camera/color/camera_info`)
- `text_prompt` - CLIP text prompt for similarity (default: `"a photo of a person"`)
- `map_frame` - Fixed frame for semantic map (default: `camera_color_optical_frame`)
- `camera_frame` - Camera frame for detections (default: `camera_color_optical_frame`)

---

### Option 2: Run Nodes in Separate Terminals

**Terminal 1: YOLO Segmentation Node**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run yolo11_seg_bringup 3d_yolo11_seg_node_main \
    --ros-args \
    -p model_path:=/home/sensor/yolov8n-seg.engine \
    -p image_topic:=/camera/camera/color/image_raw \
    -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
    -p camera_info_topic:=/camera/camera/color/camera_info \
    -p text_prompt:="a photo of a person" \
    -p conf:=0.25 \
    -p iou:=0.70 \
    -p imgsz:=640 \
    -p depth_scale:=1000.0 \
    -p pc_downsample:=2 \
    -p pc_max_range:=8.0
```

**Terminal 2: Semantic Mapper Node**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run yolo11_seg_bringup mapper_node2 \
    --ros-args \
    -p detection_message:=/yolo/detections \
    -p output_dir:=/home/sensor/ros2_ws/src/yolo11_seg_bringup \
    -p export_interval:=5.0 \
    -p map_frame:=camera_color_optical_frame \
    -p camera_frame:=camera_color_optical_frame \
    -p semantic_map_topic:=/yolo/semantic_map
```

**Terminal 3: Semantic Map Reader**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run yolo11_seg_bringup clip_reader
```

---

## Visualization

### RViz Configuration

To visualize the outputs in RViz:

1. Add display for `/yolo/pointcloud` (PointCloud2)
   - Set Color Transformer to "RGB8"
   - Adjust point size for better visibility

2. Add display for `/yolo/centroids` (MarkerArray)
   - Shows object centroids as colored spheres

3. (Optional) Add displays for `/yolo/annotated` and `/yolo/clip_boxes` (Image)
   - Enable by setting `publish_annotated:=true` and `publish_clip_boxes_vis:=true`

### Topic Monitoring

```bash
# Monitor detection rate
ros2 topic hz /yolo/detections

# Echo semantic map updates
ros2 topic echo /yolo/semantic_map

# Check pointcloud output
ros2 topic echo /yolo/pointcloud --no-arr
```

---

## Configuration

### Key Parameters

**YOLO Detection:**
- `conf` (0.0-1.0): Confidence threshold for detections (default: 0.25)
- `iou` (0.0-1.0): IOU threshold for NMS (default: 0.70)
- `imgsz` (int): Input image size for model (default: 640)
- `mask_threshold` (0.0-1.0): Segmentation mask threshold (default: 0.5)

**Pointcloud Generation:**
- `depth_scale` (float): Depth units to meters conversion (default: 1000.0 for mm)
- `pc_downsample` (int): Pointcloud downsampling factor (default: 2)
- `pc_max_range` (float): Maximum depth range in meters (default: 8.0)

**Semantic Mapping:**
- `distance_threshold` (float): Spatial merge threshold in meters (default: 0.2)
- `export_interval` (float): CSV export interval in seconds (default: 5.0)

---

## Output Files

### CSV Format (`detections.csv`, `detections_final.csv`)
```csv
class_name,x,y,z
person,1.234,2.345,0.678
chair,0.567,1.890,0.234
```

### JSON Format (`config/map.json`)
```json
[
    {
        "class": "person",
        "coords": {
            "x": 1.234,
            "y": 2.345,
            "z": 0.678
        }
    }
]
```

---

## Troubleshooting

**No detections appearing:**
- Check camera topics are publishing: `ros2 topic list`
- Verify model path is correct
- Lower confidence threshold: `-p conf:=0.15`
- Check camera_info is being received

**TF transform errors:**
- Ensure `map_frame` and `camera_frame` parameters match your TF tree
- Use `ros2 run tf2_ros tf2_echo <map_frame> <camera_frame>` to verify transforms

**Low FPS / Performance issues:**
- Increase `pc_downsample` parameter (2, 4, or 8)
- Reduce `imgsz` to 416 or 320
- Disable optional visualizations: `-p publish_annotated:=false`
- Ensure CUDA is available: Check logs for "Loading CLIP model on cuda"

**Memory issues on Jetson:**
- Reduce model size (use YOLOv8n-seg instead of larger variants)
- Increase `pc_downsample` to reduce pointcloud memory
- Lower `pc_max_range` to limit pointcloud extent

---

## Development

### Running Tests
```bash
cd ~/ros2_ws
colcon test --packages-select yolo11_seg_bringup
```

### Code Style
This package follows ROS2 Python style guidelines and includes linting tests for:
- `ament_copyright`
- `ament_flake8`
- `ament_pep257`

---

## License

Apache-2.0

## Maintainer

sensor@todo.todo
