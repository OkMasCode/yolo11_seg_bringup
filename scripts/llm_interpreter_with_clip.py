from unittest import result
import os
import json
import time
import ollama
from pydantic import BaseModel
from typing import List, Dict
import torch
import numpy as np
import sys
import gc

# Add path for CLIP processor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from yolo11_seg_bringup.utils.clip_processor import CLIPProcessor

CHAT_MODEL = "llama3.2:3b"

OLLAMA_HOST = "http://localhost:11435"

MAP_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/map.json"
CLUSTERED_MAP_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/clustered_map.json"
ROBOT_COMMAND_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json"

DICTIONARY = [
    "sink", "refrigerator", "oven", "microwave", "toaster", "dishwasher",
    "dining table", "kitchen counter", "stove", "cabinet", "sofa", "couch",
    "tv", "television", "armchair", "coffee table", "bookshelf", "lamp",
    "window", "door", "bed", "nightstand", "dresser", "closet", "mirror",
    "toilet", "bathtub", "shower", "bathroom mirror", "towel rack", "chair",
    "table", "desk", "light", "painting", "picture frame", "plant",
    "potted plant", "rug", "carpet", "curtain", "blind", "wall", "floor",
    "ceiling", "shelf", "box", "basket",
]

client = ollama.Client(host=OLLAMA_HOST)

house_map = []
clustered_map = []
cluster_summaries = []
clip_processor = None
goal_text_embedding = None

# ------------------ FUNCTIONS ----------------- #

def initialize_clip():
    """Initialize CLIP processor."""
    global clip_processor
    # Force CPU if low on memory, or use CUDA if available
    device = os.environ.get("CLIP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing CLIP processor on device: {device}\n")
    
    # Clear any existing CLIP processor to free memory
    if clip_processor is not None:
        del clip_processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    clip_processor = CLIPProcessor(
        device=device,
        model_name="ViT-B-16-SigLIP",
        pretrained="webli"
    )

def wait_for_server():
    """Ensures the Ollama container is reachable."""
    print(f"Connecting to Ollama brain at {OLLAMA_HOST}...")
    retries = 0
    while True:
        try:
            response = ollama.Client(host=OLLAMA_HOST).list()
            print("Successfully connected to Ollama!\n")
            break
        except Exception:
            retries += 1
            if retries > 30:
                print(f"Failed to connect to Ollama after {retries} attempts.")
                sys.exit(1)
            time.sleep(1)

def load_house_map(filename):
    """Loads the map.json file containing all objects and their coordinates."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Map file not found: {filename}")
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert dict structure to list format for compatibility
    if isinstance(data, dict):
        # map.json has structure: {"id": {"name": ..., "pose_map": ..., "image_embedding": ...}}
        result = []
        for obj_id, obj_data in data.items():
            obj_data["id"] = obj_id
            result.append(obj_data)
        return result
    
    return data if isinstance(data, list) else []

def load_clustered_map(filename):
    """Loads the clustered_map.json file containing clusters and outliers."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Clustered map file not found: {filename}")
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data if isinstance(data, list) else []

def summarize_clusters(clustered_map_data):
    """
    Generates simple summaries of clusters.
    Returns a dict mapping cluster_id to list of object classes.
    """
    cluster_objects = {}
    
    for entry in clustered_map_data:
        cluster_id = entry.get("cluster")
        obj_class = entry.get("class", "")
        
        if cluster_id not in cluster_objects:
            cluster_objects[cluster_id] = []
        
        cluster_objects[cluster_id].append(obj_class)
    
    return cluster_objects

def compute_cluster_coords(cluster_id: int) -> Dict | None:
    """
    Extract centroid coordinates (x, y, z) for a given cluster_id
    from the cluster_centroid field in clustered_map. Returns None if not found.
    """
    for entry in clustered_map:
        if entry.get("cluster") != cluster_id:
            continue
        centroid = entry.get("cluster_centroid")
        if isinstance(centroid, dict):
            return centroid
    
    return None

def get_map_objects():
    """Returns all unique object classes from the map."""
    return sorted(list(set(obj.get("name", "") for obj in house_map if obj.get("name"))))

def find_objects(class_name: str):
    """Find all instances of an object in the map."""
    class_name_lower = class_name.strip().lower()
    matches = []
    
    for obj in house_map:
        if obj.get("name", "").lower() == class_name_lower:
            matches.append(obj)
    
    return matches

def find_object_clusters(class_name: str):
    """
    Find which clusters contain the given object class.
    Returns a list of cluster_ids.
    """
    class_name_lower = class_name.strip().lower()
    found_cluster_ids = []
    
    for cluster_id, objects in cluster_summaries.items():
        # Check if the object is in this cluster
        objects_lower = [obj.lower() for obj in objects]
        
        if class_name_lower in objects_lower:
            found_cluster_ids.append(cluster_id)
    
    return found_cluster_ids

def compute_object_similarities(goal_objects: List[Dict], text_embedding) -> List[Dict]:
    """
    Compute CLIP similarity scores for all goal objects.
    Ranks them by similarity score (highest first).
    """
    if text_embedding is None:
        print("Warning: Text embedding is None, cannot compute similarities")
        return goal_objects
    
    scored_objects = []
    
    for obj in goal_objects:
        image_embedding = obj.get("image_embedding")
        
        if image_embedding is None:
            print(f"Warning: Object {obj.get('id')} has no image embedding")
            scored_objects.append({
                **obj,
                "similarity_score": 0.0
            })
            continue
        
        # Convert to numpy if needed
        if isinstance(image_embedding, list):
            image_embedding = np.array(image_embedding, dtype=np.float32)
        
        # Compute similarity using CLIP processor
        try:
            similarity = clip_processor.compute_sigmoid_probs(image_embedding, text_embedding)
            if similarity is None:
                similarity = 0.0
        except Exception as e:
            print(f"Error computing similarity for {obj.get('id')}: {e}")
            similarity = 0.0
        
        scored_objects.append({
            **obj,
            "similarity_score": float(similarity)
        })
        
        # Clear image embedding from memory after use
        del image_embedding
    
    # Sort by similarity score (highest first)
    scored_objects.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    
    # Clear text embedding reference
    del text_embedding
    gc.collect()
    
    return scored_objects

def find_goal_objects(goal: str, clip_prompts: List[str]):
    """
    Find goal objects and rank them by CLIP similarity.
    Uses ensemble of CLIP prompts.
    """
    print("2. .......Extracting objects in the map.........\n")

    goal_objects = find_objects(goal)

    if goal_objects:
        print(f"Found {len(goal_objects)} objects of class {goal} in the map")
        
        # Encode text prompts as ensemble
        text_embedding = clip_processor.encode_text(clip_prompts)
        
        if text_embedding is not None:
            print("Computing CLIP similarities...\n")
            # Rank by similarity
            goal_objects = compute_object_similarities(goal_objects, text_embedding)
            
            # Print ranking
            print("Ranked by similarity:")
            for i, obj in enumerate(goal_objects, 1):
                sim_score = obj.get("similarity_score", 0.0)
                print(f"  {i}. {obj.get('id')} - Similarity: {sim_score:.2f}%")
            print()
            
            # Clear CUDA cache after similarity computation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print("Warning: Could not encode text embedding\n")
    else:
        print(f"No objects of class {goal} found in the map.")
    
    return goal_objects

# --------------- OUTPUT FORMATS --------------- #

class NavResult(BaseModel):
    goal: str # class of the object to navigate to
    goal_objects: List[Dict] # list of objects of that class in the map (ranked by similarity)
    clip_prompts: List[str] # list of 3 prompts for CLIP model to find the object visually
    action: str # high-level action plan to reach the goal
    cluster_info: Dict | None # information about the most likely cluster

class Goal(BaseModel):
    goal: str # class of the object to navigate to
    clip_prompts: List[str] # list of 3 prompts for CLIP model to find the object visually

class Action(BaseModel):
    action: str # Action that the robot has to perform

class ClusterPrediction(BaseModel):
    cluster_id: int  # ID of the most likely cluster
    reasoning: str  # Brief explanation of why this cluster was chosen

# ------------------ LLM CALLS ----------------- #

def extract_goal(prompt : str) -> Goal:

    print("1. .......Extracting goal from prompt.........\n")
    SYSTEM_PROMPT = f"""You are a goal extraction system for a robot.

    VALID HOUSEHOLD OBJECTS LIST:
    {DICTIONARY}

    CRITICAL RULES - YOU MUST FOLLOW EXACTLY:

    1. GOAL EXTRACTION:
    - The 'goal' field MUST be EXACTLY one string from the valid objects list above
    - Match the EXACT string format from the list (case-sensitive)
    - Handle synonyms by mapping to the exact list entry:
        * "tv" → "television" (if "television" is in list)
        * "couch" → "sofa" (if "sofa" is in list)
        * "fridge" → "refrigerator"
    - If NO valid match exists, return goal as empty string ""
    - NEVER invent new object names
    - NEVER return objects not in the valid list

    2. CLIP PROMPTS CONSTRUCTION (Generate EXACTLY 3 variations):
    - You MUST provide EXACTLY 3 different prompt variations for the same object
    - Format: "<features> <goal>" where goal is the exact object name
    - ALWAYS include the goal object name in ALL clip_prompts
    - Generate 3 semantic variations by:
        * Reordering features and object name
        * Using synonymous phrasing for the same features
        * Emphasizing different aspects of the description
    - If user provides features (color, size, position, etc.):
        * Example: "Go to the mug with red stripes" → clip_prompts: ["mug with red stripes", "red striped mug", "red stripes on a mug"]
        * Example: "Navigate to the small black tv" → clip_prompts: ["small black television", "black small television", "television that is small and black"]
    - If NO features provided, generate variations using just the object name:
        * Example: "Go to sofa" → clip_prompts: ["sofa", "a sofa", "the sofa"]
    - DO NOT add features that were not mentioned by the user
    - DO NOT use isolated words, always include the object name in each variation
    - ALL 3 prompts must describe the SAME object with the SAME features, just phrased differently

    Output JSON: {{"goal": "<exact string from list or empty>", "clip_prompts": ["<variation 1>", "<variation 2>", "<variation 3>"]}}"""

    # message passed to the LLM
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    start_time = time.time()
    # LLM call
    respose = client.chat(
        model = CHAT_MODEL,
        messages= msgs,
        format = Goal.model_json_schema()
    )
    end_time = time.time()
    elapsed = end_time - start_time
 
    result = Goal.model_validate_json(respose["message"]["content"])
    print(f"Goal: {result.goal}")
    print(f"Clip Prompts:")
    for i, prompt in enumerate(result.clip_prompts, 1):
        print(f"  {i}. {prompt}")
    print(f"Computation time: {elapsed:.2f} seconds\n")
    return result

def determine_most_likely_cluster(prompt: str, goal: str) -> ClusterPrediction:
    """
    Uses LLM to analyze the user prompt and clustered map to determine
    the most likely cluster that contains the goal object.
    """
    print("3. .......Determining most likely cluster with LLM.........\n")
    
    # Format cluster information for the LLM
    cluster_descriptions = []
    for cluster_id, objects in cluster_summaries.items():
        cluster_descriptions.append(f"Cluster {cluster_id}: {', '.join(objects)}")
    
    clusters_text = "\n".join(cluster_descriptions)
    
    SYSTEM_PROMPT = f"""You are a spatial reasoning system for robot navigation.

    Your task: Analyze the user's navigation prompt and determine which cluster is MOST LIKELY to contain the goal object.

    AVAILABLE CLUSTERS IN THE HOUSE:
    {clusters_text}

    CLUSTER MATCHING STRATEGY:
    1. Semantic Content Analysis:
       - Examine each cluster's object contents
       - Match clusters to typical room types based on objects they contain
       - Find which cluster(s) contain the goal object
    
    2. Context-Based Refinement:
       - Analyze contextual clues in the user's prompt:
         * Room mentions ("kitchen", "bathroom", "living room", "bedroom")
         * Spatial references ("near the", "next to", "in the")
         * Activity context ("to watch TV" → cluster with TV)
         * Object associations ("bring me the remote" → cluster with TV/sofa)
    
    3. Semantic room groupings (for reference):
       - Kitchen: refrigerator, oven, microwave, stove, dishwasher, toaster, kitchen counter, cabinet, sink
       - Bathroom: toilet, bathtub, shower, sink, towel rack, bathroom mirror
       - Living room: sofa, couch, tv, television, armchair, coffee table, lamp, bookshelf
       - Bedroom: bed, nightstand, dresser, closet, mirror
       - Dining area: dining table, chair
    
    4. Selection Priority:
       - FIRST: Direct match - if goal object is directly in a cluster, strongly prefer that cluster
       - SECOND: Semantic match - if goal not found, select cluster with most semantically related objects
       - THIRD: Context match - use activity context and room mentions to refine selection
    
    5. Ambiguity Resolution:
       - If the goal object appears in multiple clusters: use contextual clues to disambiguate
       - Consider typical usage patterns
       - Prefer the cluster with more semantically related objects

    DECISION PROCESS:
    1. List which cluster(s) contain the goal object (if any)
    2. Analyze contextual clues in the prompt
    3. Select the cluster that BEST matches the context
    4. Provide brief reasoning for your choice

    CRITICAL RULES:
    - MUST select a cluster_id from the available clusters above
    - NEVER invent cluster IDs that don't exist
    - All clusters are valid choices (no outliers)
    - If goal not found in any cluster, select the cluster with most related objects
    - Reasoning should be concise (1-2 sentences)
    - Always consider the full context of the user's prompt
    - YOUR REASONING MUST MATCH YOUR SELECTED CLUSTER_ID - explain why this specific cluster is chosen
    - DO NOT recommend a different cluster in your reasoning than the one you select as cluster_id

    EXAMPLES:
    Prompt: "Go to the sofa in the living room" | Goal: "sofa" | Clusters: [Cluster 0: [sofa, tv, coffee table], Cluster 1: [bed, nightstand]]
    → {{"cluster_id": 0, "reasoning": "Sofa is in Cluster 0 which represents the living room with TV and coffee table."}}

    Prompt: "Bring me the remote from near the TV" | Goal: "remote" | Clusters: [Cluster 0: [tv, sofa], Cluster 1: [bed]]
    → {{"cluster_id": 0, "reasoning": "Remote is likely near TV, which is in Cluster 0 with sofa."}}

    Output JSON: {{"cluster_id": <int>, "reasoning": "<brief explanation>"}}"""
    
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User prompt: '{prompt}'\nGoal object: '{goal}'"}
    ]
    
    start_time = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=msgs,
        format=ClusterPrediction.model_json_schema()
    )
    end_time = time.time()
    elapsed = end_time - start_time
    
    result = ClusterPrediction.model_validate_json(response["message"]["content"])
    
    # Validate cluster_id exists
    if result.cluster_id not in cluster_summaries:
        print(f"Warning: LLM returned invalid cluster_id {result.cluster_id}, using first available cluster")
        original_cluster_id = result.cluster_id
        result.cluster_id = list(cluster_summaries.keys())[0]
        result.reasoning = f"Adjusted from invalid cluster {original_cluster_id}. {result.reasoning}"
    
    print(f"Selected Cluster: {result.cluster_id}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Computation time: {elapsed:.2f} seconds\n")
    
    return result
 
def extract_action(prompt : str) -> Action:
        
    print("4. .......Extracting action from prompt.........\n")
    SYSTEM_PROMPT = """You are a robot action interpreter for household navigation.

    Your task: Analyze the user's prompt and determine what action the robot should perform.

    ACTIONS:

    1. go_to_object: User wants the robot to NAVIGATE to an object and STAY THERE
    - Purpose: Robot goes to location and remains there
    - Keywords indicating this action:
        * "go to", "navigate to", "move to", "visit", "head to", "approach"
        * "where is the", "take me to", "go find the"
    - Examples:
        * "Go to the kitchen"
        * "Navigate to the sofa"
        * "Move to the bathroom"
        * "Take me to the bedroom"
        * "Where is the refrigerator?"

    2. bring_back_object: User wants the robot to GO to an object, TAKE/GRAB it, and BRING IT BACK
    - Purpose: Robot retrieves an object and returns with it
    - Keywords indicating this action:
        * "bring", "fetch", "get", "retrieve", "grab", "carry", "bring me", "bring back"
        * "pick up", "collect", "grab me", "get me"
    - Examples:
        * "Bring me the remote"
        * "Fetch the book from the shelf"
        * "Get me a blanket"
        * "Retrieve the phone from the kitchen"
        * "Grab me the TV remote"

    DECISION RULES:

    1. Action detection priority:
    - If user says "bring", "fetch", "get", "retrieve", "grab", "carry" → bring_back_object
    - If user says "go to", "navigate", "move", "visit", "take me to" → go_to_object
    
    2. Handle ambiguous cases:
    - If user mentions BOTH actions or unclear intent → default to go_to_object
    - If user just mentions an object without action verb → go_to_object (safest default)
    
    3. Action confirmation:
    - Double-check: Does user want robot to RETURN with object? → bring_back_object
    - Does user want robot to just GO somewhere? → go_to_object

    CRITICAL RULES:
    - ALWAYS output either "go_to_object" or "bring_back_object", never anything else
    - Choose the most obvious action from the prompt
    - If truly ambiguous, default to "go_to_object"
    - Do not overthink - use keyword matching when clear

    Output JSON: {"action": "go_to_object" or "bring_back_object"}"""
    
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    start_time = time.time()
    respose = client.chat(
        model = CHAT_MODEL,
        messages= msgs,
        format = Action.model_json_schema()
    )
    end_time = time.time()
    elapsed = end_time - start_time
 
    result = Action.model_validate_json(respose["message"]["content"])
    print(f"Action: {result.action}")
    print(f"Computation time: {elapsed:.2f} seconds\n")
    return result

# ---------------- MAIN PIPELINE --------------- #

def process_nav_instruction(prompt : str) -> NavResult:
    """ Full pipeline to process a navigation instruction. """
    print("\n" + "="*60)
    print("PROCESSING REQUEST")
    print("="*60 + "\n") 

    # 1. Extract goal from prompt
    goal = extract_goal(prompt)

    if not goal.goal:
        print("No valid goal extracted from the prompt.")
        gc.collect()
        return NavResult(
            goal="",
            goal_objects=[],
            clip_prompts=[],
            action="",
            cluster_info=None
        )
    
    # 2. Read the map and find objects of the goal class (ranked by CLIP similarity)
    goal_objects = find_goal_objects(goal.goal, goal.clip_prompts)

    # 3. Determine most likely cluster using LLM
    cluster_prediction = determine_most_likely_cluster(prompt, goal.goal)
    cluster_coords = compute_cluster_coords(cluster_prediction.cluster_id)
    cluster_info = {
        "cluster_id": cluster_prediction.cluster_id,
        "objects": cluster_summaries[cluster_prediction.cluster_id],
        "reasoning": cluster_prediction.reasoning,
        "coords": cluster_coords
    }
    
    # 4. Determine action plan
    action = extract_action(prompt)
    
    # Free memory after processing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return NavResult(
        goal=goal.goal,
        goal_objects=goal_objects,
        clip_prompts=goal.clip_prompts,
        action=action.action,
        cluster_info=cluster_info
    )

# ----------------- RESULT SAVING ----------------- #

def _objects_with_similarity(objects: List[Dict]) -> List[Dict]:
    """Reduce map objects to schema {id, coords, similarity_score}."""
    out = []
    for obj in objects:
        obj_id = obj.get("id")
        coords = obj.get("pose_map")
        similarity = obj.get("similarity_score", 0.0)
        
        if not obj_id:
            continue
        
        entry = {
            "id": obj_id,
            "similarity_score": float(similarity)
        }
        
        if isinstance(coords, dict):
            entry["coords"] = {
                "x": float(coords.get("x", 0.0)),
                "y": float(coords.get("y", 0.0)),
                "z": float(coords.get("z", 0.0))
            }
        
        out.append(entry)
    
    return out

def _build_alternatives(cluster_id: int, goal_class: str) -> List[Dict]:
    """Pick up to 3 alternative objects from the selected cluster."""
    candidates: List[Dict] = []
    # Prefer clustered_map entries to keep cluster context
    for entry in clustered_map:
        if entry.get("cluster") != cluster_id:
            continue
        cls = entry.get("class", "")
        if not cls or cls == goal_class:
            continue
        alt = {"class": cls}
        coords = entry.get("coords") or entry.get("pose_map")
        if not coords:
            continue
        
        if isinstance(coords, dict):
            alt["coords"] = coords
        
        candidates.append(alt)

    # Deduplicate by class and cap to 3
    seen = set()
    unique: List[Dict] = []
    for alt in candidates:
        cls = alt.get("class", "")
        if cls and cls not in seen:
            seen.add(cls)
            unique.append(alt)
        if len(unique) >= 3:
            break
    
    return unique

def save_robot_command(output_path: str, prompt: str, result: NavResult) -> None:
    """Serialize navigation result into robot_command.json schema and save."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    valid = bool(result.goal) and result.action in {"go_to_object", "bring_back_object"}
    cluster_info: Dict | None = None
    if result.cluster_info:
        cluster_info = {
            "cluster_id": result.cluster_info.get("cluster_id"),
            "objects": result.cluster_info.get("objects", []),
            "reasoning": result.cluster_info.get("reasoning", ""),
            "coords": result.cluster_info.get("coords"),
        }

    payload = {
        "timestamp": time.time(),
        "prompt": prompt,
        "goal": result.goal,
        "goal_objects": _objects_with_similarity(result.goal_objects),
        "cluster_info": cluster_info,
        "clip_prompts": result.clip_prompts,
        "action": result.action,
        "valid": valid,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    print(f"Saved robot command to: {output_path}")

# -------------------- MAIN -------------------- #

def main():
    global house_map, clustered_map, cluster_summaries
    wait_for_server()
    initialize_clip()

    house_map = load_house_map(MAP_FILE)
    map_objects = get_map_objects()
    
    # Load clustered map and generate summaries
    try:
        clustered_map = load_clustered_map(CLUSTERED_MAP_FILE)
        cluster_summaries = summarize_clusters(clustered_map)
        print(f"Loaded {len(clustered_map)} cluster entries")
        print(f"Generated summaries for {len(cluster_summaries)} clusters")
        print(f"Cluster breakdown:")
        for cluster_id, objects in cluster_summaries.items():
            print(f"  Cluster {cluster_id}: {', '.join(objects)}")
        print()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Continuing without cluster information\n")

    print(f"Number of objects in the house map: {len(house_map)}")
    print(f"Number of unique classes in the map: {len(map_objects)}")
    print(f"Number of recognizable classes: {len(DICTIONARY)}\n")

    while True:
        user_prompt = input("Navigation Instruction: ").strip()
        if not user_prompt:
            continue
        
        process_start = time.time()
        result = process_nav_instruction(user_prompt)
        process_end = time.time()
        
        print(f"\nTotal processing time: {process_end - process_start:.2f} seconds")
        
        # Persist the result to the configured output JSON
        save_robot_command(ROBOT_COMMAND_FILE, user_prompt, result)


if __name__ == "__main__":
    main()
