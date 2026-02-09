import os
import json
import time
from pydantic import BaseModel
from typing import List, Dict
import torch
import sys
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import warnings
import logging

# Suppress HuggingFace warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
logging.getLogger('transformers').setLevel(logging.ERROR)

# Configure Hugging Face model
# Using Mistral-7B for Jetson Thor - excellent performance, no authentication required
CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_MAX_LENGTH = 8192  # Mistral supports longer context
OFFLINE_MODE = True  # Set to True for offline operation (uses cached model)

MAP_FILE = "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/map.json"
CLUSTERED_MAP_FILE = "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/clustered_map.json"
ROBOT_COMMAND_FILE = "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json"
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

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

llm_pipeline = None

house_map = []
clustered_map = []
cluster_summaries = []

# ------------------ FUNCTIONS ----------------- #

def load_prompt(filename: str, **kwargs) -> str:
    """Load a prompt template from file and format with variables.
    
    Args:
        filename: Name of the prompt file (e.g., 'extract_goal.txt')
        **kwargs: Variables to substitute in the prompt template
    
    Returns:
        Formatted prompt string
    """
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, 'r', encoding='utf-8') as f:
        template = f.read()
    return template.format(**kwargs)

def initialize_model():
    """Initialize the Hugging Face model and tokenizer."""
    global llm_pipeline
    print(f"Loading language model: {CHAT_MODEL}")
    print(f"Device: {MODEL_DEVICE}")
    print(f"Offline mode: {OFFLINE_MODE}\n")
    
    try:
        # Load tokenizer and model
        # local_files_only=True ensures offline operation after first download
        tokenizer = AutoTokenizer.from_pretrained(
            CHAT_MODEL,
            local_files_only=OFFLINE_MODE
        )
        
        print("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            CHAT_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            local_files_only=OFFLINE_MODE
        )
        
        print("Creating inference pipeline...")
        # Create text generation pipeline
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        print("Successfully loaded language model!\n")
    except Exception as e:
        print(f"Failed to load model: {e}")
        if OFFLINE_MODE:
            print("\nIf this is your first run, set OFFLINE_MODE=False to download the model.")
            print("After download, the model will be cached locally for offline use.")
        sys.exit(1)

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

def get_cluster_dimensions(cluster_id: int) -> Dict | None:
    """
    Extract cluster dimensions (bounding_box and radius) for a given cluster_id
    from the cluster_dimensions field in clustered_map. Returns None if not found.
    """
    for entry in clustered_map:
        if entry.get("cluster") != cluster_id:
            continue
        dimensions = entry.get("cluster_dimensions")
        if isinstance(dimensions, dict):
            return dimensions
    
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

def find_goal_objects(goal: str):
    """
    Find goal objects in the map.
    """
    print("2. .......Extracting objects in the map.........\n")

    goal_objects = find_objects(goal)

    if goal_objects:
        print(f"Found {len(goal_objects)} objects of class {goal} in the map\n")
    else:
        print(f"No objects of class {goal} found in the map.\n")
    
    return goal_objects

# --------------- OUTPUT FORMATS --------------- #

class NavResult(BaseModel):
    goal: str # class of the object to navigate to
    goal_objects: List[Dict] # list of objects of that class in the map
    action: str # high-level action plan to reach the goal
    cluster_info: Dict | None # information about the most likely cluster

class Goal(BaseModel):
    goal: str # class of the object to navigate to

class Action(BaseModel):
    action: str # Action that the robot has to perform

class ClusterPrediction(BaseModel):
    cluster_id: int  # ID of the most likely cluster
    reasoning: str  # Brief explanation of why this cluster was chosen

# ------------------ LLM CALLS ----------------- #

def call_llm(messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.3) -> str:
    """Call the LLM with a list of messages and return the response.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        temperature: Lower temperature (0.1-0.3) for more deterministic JSON output
    """
    # Format messages for the model (using chat template)
    if hasattr(llm_pipeline.tokenizer, 'apply_chat_template'):
        prompt = llm_pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback formatting for Mistral
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"[INST] {content} [/INST]\n\n"
            elif role == "user":
                prompt += f"[INST] {content} [/INST]\n"
    
    # Generate response with lower temperature for more consistent JSON
    response = llm_pipeline(
        prompt,
        max_new_tokens=max_tokens,
        max_length=None,  # Explicitly set to None to avoid warning
        return_full_text=False,
        pad_token_id=llm_pipeline.tokenizer.eos_token_id,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=0.95
    )
    
    return response[0]["generated_text"].strip()

def extract_json_from_response(response: str, expected_keys: List[str] = None) -> Dict:
    """Extract and validate JSON object from LLM response.
    
    Args:
        response: Raw LLM response text
        expected_keys: List of keys that must be present in the JSON
    
    Returns:
        Parsed and validated JSON dict
    """
    # Strategy 1: Try parsing the entire response as JSON
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict):
            if expected_keys and all(key in parsed for key in expected_keys):
                return parsed
            elif not expected_keys:
                return parsed
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find JSON object using regex (nested braces support)
    # Match complete JSON objects with proper nesting
    json_patterns = [
        r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',  # Nested JSON
        r'\{[^{}]*\}',  # Simple JSON
    ]
    
    for pattern in json_patterns:
        matches = re.finditer(pattern, response, re.DOTALL)
        for match in matches:
            try:
                candidate = match.group(0)
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    # Validate expected keys if provided
                    if expected_keys:
                        if all(key in parsed for key in expected_keys):
                            return parsed
                    else:
                        return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Try to extract JSON between markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*({[^`]+})\s*```', response, re.DOTALL)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    
    # If all strategies fail, raise error with helpful message
    raise ValueError(
        f"Could not extract valid JSON from LLM response.\n"
        f"Expected keys: {expected_keys}\n"
        f"Response preview: {response[:500]}..."
    )

def extract_goal(prompt : str) -> Goal:

    print("1. .......Extracting goal from prompt.........\n")
    
    # Load prompt template from file
    SYSTEM_PROMPT = load_prompt('extract_goal.txt', DICTIONARY=DICTIONARY)

    # message passed to the LLM
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    start_time = time.time()
    # LLM call with error handling and retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"   Retry attempt {attempt + 1}/{max_retries}")
            
            response_text = call_llm(msgs, temperature=0.2)  # Low temp for consistent JSON
            
            response_json = extract_json_from_response(response_text, expected_keys=["goal"])
            result = Goal.model_validate(response_json)
            break  # Success, exit retry loop
            
        except (ValueError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                print(f"   JSON parsing failed, retrying...")
                time.sleep(1)
                continue
            else:
                print(f"ERROR: Failed to get valid JSON after {max_retries} attempts")
                print(f"Last response: {response_text}\n")
                raise
        except Exception as e:
            print(f"ERROR during LLM call: {type(e).__name__}")
            print(f"Error message: {str(e)}\n")
            raise
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Goal: {result.goal}")
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
    
    # Load prompt template from file
    SYSTEM_PROMPT = load_prompt('determine_cluster.txt', clusters_text=clusters_text)
    
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User prompt: '{prompt}'\nGoal object: '{goal}'"}
    ]
    
    start_time = time.time()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"   Retry attempt {attempt + 1}/{max_retries}")
            
            response_text = call_llm(msgs, temperature=0.2)
            
            response_json = extract_json_from_response(response_text, expected_keys=["cluster_id", "reasoning"])
            result = ClusterPrediction.model_validate(response_json)
            break
            
        except (ValueError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                print(f"   JSON parsing failed, retrying...")
                time.sleep(1)
                continue
            else:
                print(f"ERROR: Failed to get valid JSON after {max_retries} attempts")
                print(f"Last response: {response_text}\n")
                raise
        except Exception as e:
            print(f"ERROR during cluster prediction: {type(e).__name__}")
            print(f"Error: {str(e)}\n")
            raise
    end_time = time.time()
    elapsed = end_time - start_time
    
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
    
    # Load prompt template from file
    SYSTEM_PROMPT = load_prompt('extract_action.txt')
    
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    start_time = time.time()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"   Retry attempt {attempt + 1}/{max_retries}")
            
            response_text = call_llm(msgs, temperature=0.1)  # Very low temp for binary choice
            
            response_json = extract_json_from_response(response_text, expected_keys=["action"])
            result = Action.model_validate(response_json)
            break
            
        except (ValueError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                print(f"   JSON parsing failed, retrying...")
                time.sleep(1)
                continue
            else:
                print(f"ERROR: Failed to get valid JSON after {max_retries} attempts")
                print(f"Last response: {response_text}\n")
                raise
        except Exception as e:
            print(f"ERROR during action extraction: {type(e).__name__}")
            print(f"Error message: {str(e)}\n")
            raise
    end_time = time.time()
    elapsed = end_time - start_time
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
            action="",
            cluster_info=None
        )
    
    # 2. Read the map and find objects of the goal class
    goal_objects = find_goal_objects(goal.goal)

    # 3. Determine most likely cluster using LLM
    cluster_prediction = determine_most_likely_cluster(prompt, goal.goal)
    cluster_coords = compute_cluster_coords(cluster_prediction.cluster_id)
    cluster_dimensions = get_cluster_dimensions(cluster_prediction.cluster_id)
    cluster_info = {
        "cluster_id": cluster_prediction.cluster_id,
        "objects": cluster_summaries[cluster_prediction.cluster_id],
        "reasoning": cluster_prediction.reasoning,
        "coords": cluster_coords,
        "dimensions": cluster_dimensions
    }
    
    # 4. Determine action plan
    action = extract_action(prompt)

    return NavResult(
        goal=goal.goal,
        goal_objects=goal_objects,
        action=action.action,
        cluster_info=cluster_info
    )

# ----------------- RESULT SAVING ----------------- #

def _objects_with_coords(objects: List[Dict]) -> List[Dict]:
    """Reduce map objects to schema {id, coords}."""
    out = []
    for obj in objects:
        obj_id = obj.get("id")
        coords = obj.get("pose_map")
        
        if not obj_id:
            continue
        
        entry = {
            "id": obj_id
        }
        
        if isinstance(coords, dict):
            entry["coords"] = {
                "x": float(coords.get("x", 0.0)),
                "y": float(coords.get("y", 0.0)),
                "z": float(coords.get("z", 0.0))
            }
        
        out.append(entry)
    
    return out

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
            "dimensions": result.cluster_info.get("dimensions")
        }

    payload = {
        "timestamp": time.time(),
        "prompt": prompt,
        "goal": result.goal,
        "goal_objects": _objects_with_coords(result.goal_objects),
        "cluster_info": cluster_info,
        "action": result.action,
        "valid": valid,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    print(f"Saved robot command to: {output_path}")

# -------------------- MAIN -------------------- #

def main():
    global house_map, clustered_map, cluster_summaries
    initialize_model()

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
