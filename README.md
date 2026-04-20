# YOLO11 Segmentation Bringup Package

ROS2 package for real-time RGB segmentation, CLIP similarity scoring, semantic map export/visualization, and clustered map publishing.

## What Changed

This README reflects the current repository state.

Main updates versus older docs:
- The active vision node is now `pc_vision_v3.py` (RGB only, no depth subscription).
- The package exports JSON through `cpp_mapper_json_exporter_node.py` from a semantic map topic.
- Cluster preprocessing and clustered map publishing are available both as a script and as a ROS node.
- The LLM helper script is now `scripts/reduced_llm_transformers.py` (Transformers-based), not an Ollama script.
- There is currently no `launch/` directory in this package.

## What is missing

- In the LLM the model does not compare the extracted goal with what it's in the map (if extracted goal is television and the map contains TV, the output will still be television)

## Package Layout

```text
yolo11_seg_bringup/
├── yolo11_seg_bringup/
│   ├── pc_vision_v3.py
│   ├── map_points_node.py
│   ├── clustered_map_preproc_publisher_node.py
│   ├── clustered_map_points_node.py
│   ├── cpp_mapper_json_exporter_node.py
│   └── utils/
│       └── clip_processor_validator.py
├── scripts/
│   ├── map_preproc.py
│   ├── reduced_llm_transformers.py
│   └── prompts/
├── config/
│   ├── map_v6.json
│   ├── clustered_map_v6.json
│   ├── robot_command.json
│   └── clip_prompt.json
├── setup.py
└── package.xml
```

## Runtime Architecture

Typical flow:
1. `pc_vision_node_v3` publishes segmented detections and text embedding.
2. An external semantic mapper (for example C++ mapper) publishes `SemanticObjectArray`.
3. `cpp_mapper_json_exporter_node` writes `map_v6.json` from that semantic map topic.
4. `clustered_map_preproc_publisher_node` clusters `map_v6.json`, writes `clustered_map_v6.json`, and publishes clustered objects.
5. RViz helper nodes (`map_points_node`, `clustered_map_points_node`) publish marker arrays.
6. `scripts/reduced_llm_transformers.py` reads map files and writes `robot_command.json` for goal + CLIP prompt guidance.

## ROS Nodes

### 1) pc_vision_node_v3

Entry point:
- `ros2 run yolo11_seg_bringup pc_vision_node_v3`

Purpose:
- Runs YOLO segmentation + tracking on RGB images.
- Computes masked and unmasked SigLIP embeddings per detection.
- Computes similarity against the current CLIP prompt from `robot_command.json`.
- Publishes per-detection masks, embeddings, and similarity.

Subscriptions:
- `/camera/camera/color/image_raw` (`sensor_msgs/Image`) by default.

Publications:
- `/vision/detections` (`yolo11_seg_interfaces/DetectedObjectV3Array`)
- `/vision/text_embedding` (`std_msgs/Float32MultiArray`)
- `/vision/annotated_image` (`sensor_msgs/Image`, when visualization is enabled)

Important parameters:
- `image_topic` (default: `/camera/camera/color/image_raw`)
- `enable_visualization` (default: `True`)
- `model_path` (default: `/home/workspace/yoloe-26l-seg.pt`)
- `imgsz` (default: `640`)
- `conf` (default: `0.45`)
- `iou` (default: `0.35`)
- `CLIP_model_name` (default: `ViT-B-16-SigLIP`)
- `clip_pretrained` (default: `webli`)
- `robot_command_file` (default: `/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json`)
- `square_crop_scale` (default: `1.2`)
- `masked_score_weight` (default: `0.85`)
- `unmasked_score_weight` (default: `0.15`)
- `prompt_publish_interval` (default: `5.0` seconds)

Notes:
- CLIP text prompt source key is `clip_prompts` inside `robot_command.json`.
- This node does not subscribe to depth or camera info in its current version.

### 2) cpp_mapper_json_exporter_node

Entry point:
- `ros2 run yolo11_seg_bringup cpp_mapper_json_exporter_node`

Purpose:
- Subscribes to semantic map topic and periodically exports a JSON map file.

Subscription:
- `/vision/semantic_map_v5` (`yolo11_seg_interfaces/SemanticObjectArray`) by default.

Output file:
- `/workspaces/ros2_ws/src/yolo11_seg_bringup/config/map_v6.json`

Important parameters:
- `input_topic` (default: `/vision/semantic_map_v5`)
- `output_dir` (default: `/workspaces/ros2_ws/src/yolo11_seg_bringup/config`)
- `output_map_file` (default: `map_v6.json`)
- `export_interval` (default: `3.0` seconds)

### 3) clustered_map_preproc_publisher_node

Entry point:
- `ros2 run yolo11_seg_bringup clustered_map_preproc_publisher_node`

Purpose:
- Loads map JSON, clusters objects with DBSCAN, writes clustered JSON, and publishes clustered map messages.

Publication:
- `/vision/clustered_map_v6` (`yolo11_seg_interfaces/ClusteredMapObjectArray`) by default.

Files:
- Input: `/workspaces/ros2_ws/src/yolo11_seg_bringup/config/map_v6.json`
- Output: `/workspaces/ros2_ws/src/yolo11_seg_bringup/config/clustered_map_v6.json`

Important parameters:
- `input_map_file`
- `output_clustered_map_file`
- `output_topic` (default: `/vision/clustered_map_v6`)
- `frame_id` (default: `map`)
- `eps` (default: `2.4`)
- `min_samples` (default: `2`)
- `publish_rate_hz` (default: `0.5`)

### 4) map_points_node

Entry point:
- `ros2 run yolo11_seg_bringup map_points_node`

Purpose:
- Visualizes raw map objects from `map_v6.json` in RViz as point, label, and 3D box wireframe markers.

Publications:
- `/vision/map_objects_markers` (`visualization_msgs/MarkerArray`)
- `/vision/map_objects_bbox_markers` (`visualization_msgs/MarkerArray`)

Important parameters:
- `map_file` (default: `/workspaces/ros2_ws/src/yolo11_seg_bringup/config/map_v6.json`)
- `map_frame` (default: `map`)
- `marker_topic` (default: `/vision/map_objects_markers`)
- `bbox_marker_topic` (default: `/vision/map_objects_bbox_markers`)
- `publish_rate_hz` (default: `1.0`)

### 5) clustered_map_points_node

Entry point:
- `ros2 run yolo11_seg_bringup clustered_map_points_node`

Purpose:
- Visualizes clustered map entries from `clustered_map_v6.json` in RViz.

Publication:
- `/vision/clustered_map_objects_markers` (`visualization_msgs/MarkerArray`)

Important parameters:
- `clustered_map_file` (default: `/workspaces/ros2_ws/src/yolo11_seg_bringup/config/clustered_map_v6.json`)
- `map_frame` (default: `map`)
- `marker_topic` (default: `/vision/clustered_map_objects_markers`)
- `publish_rate_hz` (default: `1.0`)

## Non-ROS Scripts

### scripts/map_preproc.py

Purpose:
- Offline DBSCAN clustering from `map_v6.json` to `clustered_map_v6.json`.

Features:
- Reassigns outliers to unique cluster IDs.
- Computes centroid and 2D cluster geometry (bounding box + radius).
- Includes verbose debug prints for each stage.

Run:
```bash
cd /workspaces/ros2_ws/src/yolo11_seg_bringup
python3 scripts/map_preproc.py --eps 2.4 --min-samples 2
```

### scripts/reduced_llm_transformers.py

Purpose:
- Interactive instruction parser that uses a local Hugging Face chat model.
- Produces `config/robot_command.json` with goal class, CLIP prompt, action, and cluster reasoning.

Reads:
- `config/map_v6.json`
- `config/clustered_map_v6.json`
- Prompt templates in `scripts/prompts/`

Writes:
- `config/robot_command.json`

Current behavior highlights:
- Uses a single prompt field named `clip_prompts` in output.
- Performs multiple LLM stages (goal extraction, cluster inference, action, logic).
- Includes retry logic and JSON parsing safeguards.

Run:
```bash
cd /workspaces/ros2_ws/src/yolo11_seg_bringup
python3 scripts/reduced_llm_transformers.py
```

Model note:
- Default model is `meta-llama/Llama-3.1-8B-Instruct`.
- `OFFLINE_MODE = True` by default. Ensure model files are already cached locally, or temporarily set offline mode to `False` for first download.

## Config Files

### map_v6.json
Dictionary keyed by object ID. Typical per-object fields include:
- `name`, `frame`, `timestamp`
- `pose_map`
- `bbox_type`, `box_size`, `bbox_orientation`, `bbox_corners`
- `occurrences`
- `similarity`
- `image_embedding_masked`, `image_embedding_unmasked`
- `confidence`, `embedding_confidence`

### clustered_map_v6.json
List of objects with cluster annotations. Typical fields include:
- `id`, `cluster`, `class`, `similarity`
- `coords`
- `cluster_centroid`
- `cluster_dimensions` (2D bounding box plus radius)

### robot_command.json
Generated by the LLM script. Typical fields:
- `timestamp`, `prompt`
- `goal`
- `clip_prompts` (single string used by vision node)
- `cluster_info`
- `action`, `logic`

## Build And Install

From workspace root:

```bash
cd /workspaces/ros2_ws
colcon build --packages-select yolo11_seg_interfaces yolo11_seg_bringup
source install/setup.bash
```

Recommended Python packages (in your active environment):

```bash
pip install ultralytics torch torchvision open-clip-torch opencv-python pillow numpy scikit-learn transformers pydantic
```

## Quick Start

Terminal 1: vision
```bash
source /workspaces/ros2_ws/install/setup.bash
ros2 run yolo11_seg_bringup pc_vision_node_v3 \
  --ros-args \
  -p model_path:=/workspaces/yoloe-26l-seg.pt \
  -p image_topic:=/camera/camera/color/image_raw
```

Terminal 2: map exporter
```bash
source /workspaces/ros2_ws/install/setup.bash
ros2 run yolo11_seg_bringup cpp_mapper_json_exporter_node
```

Terminal 3: clustering publisher
```bash
source /workspaces/ros2_ws/install/setup.bash
ros2 run yolo11_seg_bringup clustered_map_preproc_publisher_node
```

Optional RViz helpers:
```bash
source /workspaces/ros2_ws/install/setup.bash
ros2 run yolo11_seg_bringup map_points_node
ros2 run yolo11_seg_bringup clustered_map_points_node
```

Optional command generation:
```bash
cd /workspaces/ros2_ws/src/yolo11_seg_bringup
python3 scripts/reduced_llm_transformers.py
```

## Known Caveats

- The package currently has no committed launch files, even though `setup.py` still includes launch file installation logic.
- `pc_vision_node_v3` class name is `NoPCVisionNode`; this is expected in the current code.
- The default model path in the node is `/home/workspace/yoloe-26l-seg.pt`; update it for your machine as needed.

## License

Apache-2.0
