import os
import json
import time
from pydantic import BaseModel
from typing import List, Dict
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import warnings
import logging

# Suppress HuggingFace warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
logging.getLogger('transformers').setLevel(logging.ERROR)

# Configure Hugging Face model
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct" #"mistralai/Mistral-7B-Instruct-v0.3"
MODEL_MAX_LENGTH = 8192
OFFLINE_MODE = True

MAP_FILE = "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/map_v6.json"
CLUSTERED_MAP_FILE = "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/clustered_map_v6.json"
ROBOT_COMMAND_FILE = "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json"
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

llm_pipeline = None

house_map = []
clustered_map = []
cluster_summaries = []
cluster_labels = {}

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
    if not kwargs:
        return template
    class _SafeFormatDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return template.format_map(_SafeFormatDict(kwargs))

def initialize_model():
    """Initialize the Hugging Face model and tokenizer."""
    global llm_pipeline
    print(f"Loading language model: {CHAT_MODEL}")
    print(f"Offline mode: {OFFLINE_MODE}\n")
    try:
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
    if isinstance(data, dict):
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

def get_cluster_entries_with_ids() -> Dict[int, List[Dict[str, str]]]:
    """Return per-cluster object candidates with stable {id, class} entries."""
    entries_by_cluster: Dict[int, List[Dict[str, str]]] = {}
    seen_ids = set()
    for entry in clustered_map:
        obj_id = str(entry.get("id", "")).strip()
        obj_class = str(entry.get("class", "")).strip()
        cluster_id = entry.get("cluster")
        if not obj_id or cluster_id is None:
            continue
        key = (cluster_id, obj_id)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        if cluster_id not in entries_by_cluster:
            entries_by_cluster[cluster_id] = []
        entries_by_cluster[cluster_id].append({
            "id": obj_id,
            "class": obj_class
        })
    for cluster_id in entries_by_cluster:
        entries_by_cluster[cluster_id] = sorted(entries_by_cluster[cluster_id], key=lambda x: x["id"])
    return entries_by_cluster

def get_labeled_cluster_descriptions_with_ids() -> List[str]:
    """Build cluster descriptions that include candidate object IDs for anchor selection."""
    descriptions = []
    entries_by_cluster = get_cluster_entries_with_ids()
    for cluster_id in sorted(cluster_summaries.keys()):
        label = cluster_labels.get(cluster_id, "unknown")
        objects = entries_by_cluster.get(cluster_id, [])
        object_text = ", ".join(f"{obj['id']}:{obj['class']}" for obj in objects)
        descriptions.append(f"Cluster {cluster_id} ({label}): {object_text}")
    return descriptions

def _cluster_object_ids(cluster_id: int | None) -> set[str]:
    """Return all object IDs belonging to the given cluster."""
    if cluster_id is None:
        return set()
    return {
        str(entry.get("id", "")).strip()
        for entry in clustered_map
        if entry.get("cluster") == cluster_id and str(entry.get("id", "")).strip()
    }

def _object_class_from_id(object_id: str) -> str:
    """Return object class for object id, empty string when unknown."""
    for entry in clustered_map:
        if entry.get("id") == object_id:
            return str(entry.get("class", "")).strip()
    return ""

def assign_cluster_labels() -> None:
    """
    Use LLM once to assign semantic names (kitchen/bedroom/etc.) to clusters.
    The labels are reused by all later LLM calls that consume cluster context.
    """
    global cluster_labels
    if not cluster_summaries:
        cluster_labels = {}
        return
    print("0. .......Assigning semantic labels to clusters.........\n")
    raw_clusters = []
    for cluster_id, objects in cluster_summaries.items():
        raw_clusters.append({"cluster_id": cluster_id, "objects": objects})
    SYSTEM_PROMPT = load_prompt('label_clusters.txt')
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"clusters": raw_clusters}, ensure_ascii=False)}
    ]
    start_time = time.time()
    max_retries = 3
    assigned = {}
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"   Retry attempt {attempt + 1}/{max_retries}")
            response_text = call_llm(msgs, temperature=0.4)
            response_json = extract_json_from_response(response_text, expected_keys=["cluster_labels"])
            labels_payload = response_json.get("cluster_labels", [])
            if not isinstance(labels_payload, list):
                labels_payload = []
            for entry in labels_payload:
                if not isinstance(entry, dict):
                    continue
                raw_id = entry.get("cluster_id")
                raw_label = entry.get("label", "")
                try:
                    cluster_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if cluster_id not in cluster_summaries:
                    continue
                label = str(raw_label).strip().lower()
                if not label:
                    continue
                assigned[cluster_id] = label
            break
        except (ValueError, json.JSONDecodeError):
            if attempt < max_retries - 1:
                print("   JSON parsing failed, retrying...")
                time.sleep(1)
                continue
            print(f"WARNING: Failed to assign labels via LLM after {max_retries} attempts. Using defaults.")
        except Exception as e:
            print(f"WARNING: Error during cluster labeling: {type(e).__name__}: {str(e)}")
            break
    cluster_labels = {}
    raw_labels = {}
    for cluster_id in cluster_summaries.keys():
        raw_labels[cluster_id] = assigned.get(cluster_id, f"cluster_{cluster_id}")
    label_counts = {}
    for label in raw_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    label_counters = {}
    for cluster_id in sorted(raw_labels.keys()):
        label = raw_labels[cluster_id]
        if label_counts[label] > 1:
            label_counters[label] = label_counters.get(label, 0) + 1
            cluster_labels[cluster_id] = f"{label} #{label_counters[label]}"
        else:
            cluster_labels[cluster_id] = label
    elapsed = time.time() - start_time
    print("Cluster labels:")
    for cluster_id in sorted(cluster_labels.keys()):
        print(f"  Cluster {cluster_id} -> {cluster_labels[cluster_id]}")
    print(f"Computation time: {elapsed:.2f} seconds\n")

def get_map_objects():
    """Returns all unique object classes from the map."""
    return sorted(list(set(obj.get("name", "") for obj in house_map if obj.get("name"))))

def _normalize_object_label(label: str) -> str:
    """Normalize an object label for case/spacing-only matching."""
    value = label.strip().lower()
    return re.sub(r"\s+", " ", value)

# --------------- OUTPUT FORMATS --------------- #

class NavResult(BaseModel):
    goal: str # class of the object to navigate to
    #goal_objects: List[Dict] # list of objects of that class in the map
    clip_prompts: str # single CLIP prompt describing the object
    anchor_object_id: str = "" # object id used as spatial anchor for navigation
    anchor_object_class: str = "" # class name for the selected anchor object id
    object_similarities: List[Dict] = [] # similarity scores for each goal object
    action: str # high-level action plan to reach the goal
    logic: str 
    cluster_info: Dict | None = None # information about the most likely cluster

class Goal(BaseModel):
    goal: str # class of the object to navigate to
    clip_prompts: str # single CLIP prompt describing the object

class Action(BaseModel):
    action: str # Action that the robot has to perform

class Logic_decision(BaseModel):
    logic: str

class ClusterPrediction(BaseModel):
    cluster_id: int | None = None  # ID of the most likely cluster, None when uncertain
    reasoning: str  # Brief explanation of why this cluster was chosen
    anchor_object_id: str = ""  # Object ID used as anchor for guiding the goal search
    location_confidence: float = 0.0  # 0.0-1.0 confidence that location cues should be prioritized

# ------------------ LLM CALLS ----------------- #

def call_llm(messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7) -> str:
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

    print("1. .......Extracting goal and CLIP prompt from prompt.........\n")
    
    # Load prompt template from file
    SYSTEM_PROMPT = load_prompt('extract_goal_and_clip.txt', MAP_OBJECTS=get_map_objects())
    # message passed to the LLM
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    start_time = time.time()
    # LLM call with error handling and retry logic
    max_retries = 3
    goal_text = ""
    clip_prompt_text = ""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"   Retry attempt {attempt + 1}/{max_retries}")
            response_text = call_llm(msgs, temperature=0.2)  # Low temp for consistent JSON
            response_json = extract_json_from_response(response_text, expected_keys=["goal", "clip_prompt"])
            goal_text = response_json["goal"]
            clip_prompt_text = response_json.get("clip_prompt", "")
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
    goal_text = goal_text.strip()
    clip_prompt_text = clip_prompt_text.strip()
    if not clip_prompt_text and goal_text:
        clip_prompt_text = goal_text
    print(f"Goal: {goal_text}")
    print(f"CLIP prompt: {clip_prompt_text}")
    print(f"Computation time: {elapsed:.2f} seconds\n")
    # Combine goal and clip prompt into the result
    result = Goal(
        goal=goal_text, 
        clip_prompts=clip_prompt_text,
    )
    return result

def determine_most_likely_cluster(prompt: str, goal: str) -> ClusterPrediction:
    """
    Uses LLM to analyze the user prompt and clustered map to determine
    the most likely cluster that contains the goal object.
    """
    print("3. .......Determining most likely cluster with LLM.........\n")
    
    # Format cluster information for the LLM
    cluster_descriptions = get_labeled_cluster_descriptions_with_ids()
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
            response_text = call_llm(msgs, temperature=0.4)
            response_json = extract_json_from_response(response_text, expected_keys=["reasoning"])
            if "location_confidence" not in response_json:
                response_json["location_confidence"] = 0.0
            if "anchor_object_id" not in response_json:
                response_json["anchor_object_id"] = ""
            # cluster_id is optional: None means "no confident cluster".
            raw_cluster_id = response_json.get("cluster_id", None)
            if raw_cluster_id in ("", "null", "None"):
                response_json["cluster_id"] = None
            elif raw_cluster_id is None:
                response_json["cluster_id"] = None
            else:
                try:
                    response_json["cluster_id"] = int(raw_cluster_id)
                except (TypeError, ValueError):
                    response_json["cluster_id"] = None
            raw_anchor_object_id = response_json.get("anchor_object_id", "")
            if raw_anchor_object_id is None:
                raw_anchor_object_id = ""
            response_json["anchor_object_id"] = str(raw_anchor_object_id).strip()
            # Clamp to [0.0, 1.0] for robustness
            try:
                response_json["location_confidence"] = float(response_json["location_confidence"])
            except (TypeError, ValueError):
                response_json["location_confidence"] = 0.0
            response_json["location_confidence"] = max(0.0, min(1.0, response_json["location_confidence"]))
            # Anchor id must belong to selected cluster. If not, clear it.
            if response_json["cluster_id"] is None:
                response_json["anchor_object_id"] = ""
            else:
                valid_ids = _cluster_object_ids(response_json["cluster_id"])
                if response_json["anchor_object_id"] not in valid_ids:
                    response_json["anchor_object_id"] = ""
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
    # Validate cluster_id exists when provided
    if result.cluster_id is not None and result.cluster_id not in cluster_summaries:
        print(f"Warning: LLM returned invalid cluster_id {result.cluster_id}. Clearing cluster selection.")
        result.cluster_id = None
        result.anchor_object_id = ""
    anchor_object_class = _object_class_from_id(result.anchor_object_id)
    print(f"Selected Cluster: {result.cluster_id if result.cluster_id is not None else 'NONE'}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Anchor object ID: {result.anchor_object_id if result.anchor_object_id else 'NONE'}")
    print(f"Anchor object class: {anchor_object_class if anchor_object_class else 'NONE'}")
    print(f"Location confidence: {result.location_confidence:.2f}")
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

def decide_logic(prompt : str) -> Logic_decision:
        
    print("5. .......Deciding logic.........\n")
    
    # Load prompt template from file
    SYSTEM_PROMPT = load_prompt('decide_logic.txt')
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
            response_json = extract_json_from_response(response_text, expected_keys=["logic"])
            result = Logic_decision.model_validate(response_json)
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
            print(f"ERROR during logic decision: {type(e).__name__}")
            print(f"Error message: {str(e)}\n")
            raise
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Logic: {result.logic}")
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
    effective_goal = goal.goal.strip() if goal.goal else ""
    if not effective_goal and goal.clip_prompts:
        effective_goal = _normalize_object_label(goal.clip_prompts)
        print(f"No explicit goal returned. Using CLIP prompt as goal hint: {effective_goal}")
    # 4. Determine most likely cluster using LLM
    cluster_prediction = determine_most_likely_cluster(prompt, effective_goal)
    anchor_object_class = _object_class_from_id(cluster_prediction.anchor_object_id)
    cluster_info = None
    if cluster_prediction.cluster_id is not None and cluster_prediction.cluster_id in cluster_summaries:
        cluster_info = {
            "cluster_id": cluster_prediction.cluster_id,
            "cluster_label": cluster_labels.get(cluster_prediction.cluster_id, "unknown"),
            "objects": cluster_summaries[cluster_prediction.cluster_id],
            "reasoning": cluster_prediction.reasoning,
            "anchor_object_id": cluster_prediction.anchor_object_id,
            "anchor_object_class": anchor_object_class,
            "location_confidence": cluster_prediction.location_confidence,
        }
    else:
        cluster_info = {
            "cluster_id": None,
            "cluster_label": "",
            "objects": [],
            "reasoning": cluster_prediction.reasoning,
            "anchor_object_id": cluster_prediction.anchor_object_id,
            "anchor_object_class": anchor_object_class,
            "location_confidence": cluster_prediction.location_confidence,
            "prioritize_location": False,
        }
    # 5. Determine action plan
    action = extract_action(prompt)
    logic = decide_logic(prompt)
    # Preserve the extracted user-intent goal even if no mapped object is selected.
    final_goal_class = effective_goal
    return NavResult(
        goal=final_goal_class,
        clip_prompts=goal.clip_prompts,
        anchor_object_id=cluster_prediction.anchor_object_id,
        anchor_object_class=anchor_object_class,
        action=action.action,
        logic=logic.logic,
        cluster_info=cluster_info,
    )

# ----------------- RESULT SAVING ----------------- #

def save_robot_command(output_path: str, prompt: str, result: NavResult) -> None:
    """Serialize navigation result into robot_command.json schema and save."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cluster_info: Dict | None = None
    if result.cluster_info:
        cluster_info = {
            "cluster_id": result.cluster_info.get("cluster_id"),
            "objects": result.cluster_info.get("objects", []),
            "reasoning": result.cluster_info.get("reasoning", ""),
            "anchor_object_id": result.cluster_info.get("anchor_object_id", ""),
            "anchor_object_class": result.cluster_info.get("anchor_object_class", ""),
            "location_confidence": float(result.cluster_info.get("location_confidence", 0.0)),
            "prioritize_location": bool(result.cluster_info.get("prioritize_location", False)),
        }
    payload = {
        "timestamp": time.time(),
        "prompt": prompt,
        "goal": result.goal,
        "clip_prompts": result.clip_prompts,
        "anchor_object_id": result.anchor_object_id,
        "anchor_object_class": result.anchor_object_class,
        "cluster_info": cluster_info,
        "action": result.action,
        "logic": result.logic,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    print(f"Saved robot command to: {output_path}")

# -------------------- MAIN -------------------- #

def main():
    global house_map, clustered_map, cluster_summaries, clip_processor
    initialize_model()
    house_map = load_house_map(MAP_FILE)
    map_objects = get_map_objects()
    # Load clustered map and generate summaries
    try:
        clustered_map = load_clustered_map(CLUSTERED_MAP_FILE)
        cluster_summaries = summarize_clusters(clustered_map)
        assign_cluster_labels()
        print(f"Loaded {len(clustered_map)} cluster entries")
        print(f"Generated summaries for {len(cluster_summaries)} clusters")
        print(f"Cluster breakdown:")
        for cluster_id, objects in cluster_summaries.items():
            print(f"  Cluster {cluster_id} ({cluster_labels.get(cluster_id, 'unknown')}): {', '.join(objects)}")
        print()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Continuing without cluster information\n")
    print(f"Number of objects in the house map: {len(house_map)}")
    print(f"Number of unique classes in the map: {len(map_objects)}")
    print(f"Goal extraction can target all classes in the loaded map\n")
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
