import os
import json
import time
import numpy as np
from pydantic import BaseModel
from typing import List, Dict
import torch
import sys
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import warnings
import logging

from yolo11_seg_bringup.utils.clip_processor_validator import CLIPProcessorValidator

# Suppress HuggingFace warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
logging.getLogger('transformers').setLevel(logging.ERROR)

# Configure Hugging Face model
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct" #"mistralai/Mistral-7B-Instruct-v0.3"
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_MAX_LENGTH = 8192
OFFLINE_MODE = False

PRINT_ALL_MAP_SIMILARITIES_TEST = True

MAP_FILE = "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/map_v6.json"
CLUSTERED_MAP_FILE = "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/clustered_map_v5.json"
ROBOT_COMMAND_FILE = "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json"
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

llm_pipeline = None
clip_processor = None

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
    print(f"Device: {MODEL_DEVICE}")
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

def initialize_clip_processor():
    """Initialize the CLIPProcessor for text and image embedding."""
    global clip_processor
    try:
        print("Initializing SigLIP CLIP processor...")
        clip_processor = CLIPProcessorValidator(
            device=MODEL_DEVICE,
            model_name='ViT-B-16-SigLIP',
            pretrained='webli',
        )
        print("Successfully loaded SigLIP CLIP processor!\n")
    except Exception as e:
        print(f"Failed to load CLIP processor: {e}")
        sys.exit(1)


def _build_validator_prompt_ensemble(clip_prompt: str) -> List[str]:
    """Expand one LLM prompt using validator-style prompt templates."""
    if not isinstance(clip_prompt, str):
        return []

    clean = clip_prompt.strip()
    if not clean:
        return []

    expanded = clip_processor.build_prompt_list(clean)
    if not expanded:
        return []

    unique = list(dict.fromkeys(expanded))
    return unique

def encode_clip_prompt(clip_prompt: str) -> np.ndarray | None:
    """
    Encode one CLIP prompt into a single averaged embedding using validator expansion.
    
    Args:
        clip_prompt: One LLM-extracted text prompt
    
    Returns:
        Normalized embedding vector (numpy array) or None if encoding fails
    """
    if not clip_prompt or clip_processor is None:
        return None

    try:
        prompt_ensemble = _build_validator_prompt_ensemble(clip_prompt)
        if not prompt_ensemble:
            return None

        print(f"Encoding {len(prompt_ensemble)} validator-style CLIP prompts")
        text_embedding = clip_processor.encode_text(prompt_ensemble)
        if text_embedding is not None:
            if not isinstance(text_embedding, np.ndarray):
                text_embedding = np.array(text_embedding)
        return text_embedding
    except Exception as e:
        print(f"Warning: Error encoding CLIP prompts: {e}")
        print(f"Continuing without text embedding...")
        return None

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

        if isinstance(image_embedding, list):
            image_embedding = np.array(image_embedding, dtype=np.float32)

        try:
            similarity = clip_processor.compute_match_score(image_embedding, text_embedding)
            if similarity is None:
                similarity = 0.0
        except Exception as e:
            print(f"Error computing similarity for {obj.get('id')}: {e}")
            similarity = 0.0
        
        scored_objects.append({
            **obj,
            "similarity_score": float(similarity)
        })

        del image_embedding

    scored_objects.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

    del text_embedding
    gc.collect()
    
    return scored_objects

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

def get_labeled_cluster_descriptions() -> List[str]:
    """Build cluster descriptions with semantic labels for downstream LLM calls."""
    descriptions = []
    for cluster_id, objects in cluster_summaries.items():
        label = cluster_labels.get(cluster_id, "unknown")
        descriptions.append(f"Cluster {cluster_id} ({label}): {', '.join(objects)}")
    return descriptions

# NOT NECESSARY
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

def _normalize_object_label(label: str) -> str:
    """Normalize an object label for case/spacing-only matching."""
    value = label.strip().lower()
    return re.sub(r"\s+", " ", value)

def find_objects(class_name: str):
    """Find all instances of an object in the map."""
    class_name_norm = _normalize_object_label(class_name)
    matches = []
    relaxed_matches = []
    
    for obj in house_map:
        object_name = obj.get("name", "")
        object_name_norm = _normalize_object_label(object_name)

        if object_name_norm == class_name_norm:
            matches.append(obj)
            continue

        # Fallback for labels like "kitchen table" vs "table"
        if class_name_norm and (
            class_name_norm in object_name_norm
            or object_name_norm in class_name_norm
        ):
            relaxed_matches.append(obj)

    if matches:
        return matches
    
    return relaxed_matches

def find_goal_objects(goal: str, clip_prompt: str) -> tuple[List[Dict], np.ndarray | None]:
    """
    Find goal objects in the map and rank them by CLIP similarity.
    Uses validator-style ensemble expansion from one CLIP prompt.
    """
    print("2. .......Extracting objects in the map.........")

    goal_objects = find_objects(goal)

    if not goal_objects:
        print(f"No map objects matched extracted goal '{goal}'. Skipping object-level map selection.\n")
        return [], None

    if goal_objects:
        print(f"Evaluating {len(goal_objects)} candidate objects for goal '{goal}'")
        
        # Encode one prompt as a validator-style ensemble.
        text_embedding = encode_clip_prompt(clip_prompt)
        
        if text_embedding is not None:
            print("Computing CLIP similarities...\n")

            if PRINT_ALL_MAP_SIMILARITIES_TEST:
                all_objects_scored = compute_object_similarities(house_map, np.copy(text_embedding))
                print("All map objects similarity (test, high precision):")
                for i, obj in enumerate(all_objects_scored, 1):
                    sim_score = obj.get("similarity_score", 0.0)
                    print(f"  {i}. {obj.get('id')} - Similarity: {sim_score:.12f}%")
                print()

            goal_objects = compute_object_similarities(goal_objects, text_embedding)

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
        print("No objects available in the map.\n")
    
    return goal_objects, text_embedding

# --------------- OUTPUT FORMATS --------------- #

class NavResult(BaseModel):
    goal: str # class of the object to navigate to
    #related_object: str = "" # object related to the goal location, empty if none
    goal_objects: List[Dict] # list of objects of that class in the map
    clip_prompts: str # single CLIP prompt describing the object
    text_embedding: List[float] | None = None # CLIP text embedding for the prompts
    object_similarities: List[Dict] = [] # similarity scores for each goal object
    action: str # high-level action plan to reach the goal
    logic: str 
    cluster_info: Dict | None = None # information about the most likely cluster
    selected_goal: Dict | None = None # final selected object, or None if no match

class Goal(BaseModel):
    goal: str # class of the object to navigate to
    #related_object: str = "" # object related to the goal location, empty if none
    clip_prompts: str # single CLIP prompt describing the object
    text_embedding: List[float] | None = None # CLIP text embedding for the prompts

class Action(BaseModel):
    action: str # Action that the robot has to perform

class Logic_decision(BaseModel):
    logic: str

class ClipPromptOutput(BaseModel):
    clip_prompt: str = ""  # Single prompt used as seed for validator expansion

#class RelatedObjectOutput(BaseModel):
#    related_object: str = ""  # related object for location grounding

class ClusterPrediction(BaseModel):
    cluster_id: int | None = None  # ID of the most likely cluster, None when uncertain
    reasoning: str  # Brief explanation of why this cluster was chosen
    location_confidence: float = 0.0  # 0.0-1.0 confidence that location cues should be prioritized

class GoalSelection(BaseModel):
    selected_object_id: str = ""  # chosen object id, empty means no match
    decision_basis: str  # "location", "similarity", "mixed", or "none"
    reasoning: str  # brief explanation

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
    SYSTEM_PROMPT = load_prompt('extract_goal.txt', MAP_OBJECTS=get_map_objects())

    # message passed to the LLM
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    start_time = time.time()
    # LLM call with error handling and retry logic
    max_retries = 3
    goal_text = ""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"   Retry attempt {attempt + 1}/{max_retries}")
            
            response_text = call_llm(msgs, temperature=0.2)  # Low temp for consistent JSON
            
            response_json = extract_json_from_response(response_text, expected_keys=["goal"])
            goal_text = response_json["goal"]
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
    print(f"Goal: {goal_text}")
    print(f"Computation time: {elapsed:.2f} seconds\n")
    
    # Now generate one CLIP prompt for this goal
    clip_prompts_result = extract_clip_prompt(prompt, goal_text)
    #related_object_result = extract_related_object(prompt, goal_text)
    
    # Combine goal and clip prompts into the result (text_embedding will be computed later in find_goal_objects)
    result = Goal(
        goal=goal_text, 
        #related_object=related_object_result.related_object,
        clip_prompts=clip_prompts_result.clip_prompt,
        text_embedding=None  # Will be computed when finding objects
    )
    
    return result

def extract_clip_prompt(prompt: str, goal: str) -> ClipPromptOutput:
    """
    Uses LLM to generate one CLIP seed prompt that describes the object
    based only on features mentioned in the user's prompt.
    The final embedding ensemble is created by validator template expansion.
    """
    print("1b. .......Generating CLIP prompt for visual matching.........\n")
    
    SYSTEM_PROMPT = load_prompt('extract_clip_prompts.txt')
    
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
            
            response_json = extract_json_from_response(response_text, expected_keys=["clip_prompt"])
            result = ClipPromptOutput.model_validate(response_json)
            result.clip_prompt = result.clip_prompt.strip()
            if not result.clip_prompt:
                raise ValueError("LLM returned empty clip_prompt")
            
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
            print(f"ERROR during CLIP prompt generation: {type(e).__name__}")
            print(f"Error message: {str(e)}\n")
            raise
    
    end_time = time.time()
    elapsed = end_time - start_time
    print("CLIP prompt generated:")
    print(f"  {result.clip_prompt}")
    print(f"Computation time: {elapsed:.2f} seconds\n")
    
    return result

# def extract_related_object(prompt: str, goal: str) -> RelatedObjectOutput:
#     """
#     Uses LLM to extract the object that appears in relation with the goal.
#     Returns empty related_object when the prompt has no location relation.
#     """
#     print("1c. .......Extracting related object from prompt.........\n")

#     cluster_descriptions = get_labeled_cluster_descriptions()
#     clusters_text = "\n".join(cluster_descriptions) if cluster_descriptions else "No cluster data available."

#     SYSTEM_PROMPT = load_prompt('extract_related_object.txt', clusters_text=clusters_text)

#     msgs = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": f"User prompt: '{prompt}'\nGoal object: '{goal}'"}
#     ]

#     start_time = time.time()
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             if attempt > 0:
#                 print(f"   Retry attempt {attempt + 1}/{max_retries}")

#             response_text = call_llm(msgs, temperature=0.2)
#             response_json = extract_json_from_response(response_text, expected_keys=["related_object"])
#             result = RelatedObjectOutput.model_validate(response_json)

#             related = result.related_object.strip()
#             if related and _normalize_object_label(related) == _normalize_object_label(goal):
#                 related = ""
#             result.related_object = related
#             break

#         except (ValueError, json.JSONDecodeError):
#             if attempt < max_retries - 1:
#                 print("   JSON parsing failed, retrying...")
#                 time.sleep(1)
#                 continue
#             print(f"ERROR: Failed to get valid JSON after {max_retries} attempts")
#             print(f"Last response: {response_text}\n")
#             raise
#         except Exception as e:
#             print(f"ERROR during related-object extraction: {type(e).__name__}")
#             print(f"Error message: {str(e)}\n")
#             raise

#     end_time = time.time()
#     elapsed = end_time - start_time
#     print(f"Related object: {result.related_object if result.related_object else 'NONE'}")
#     print(f"Computation time: {elapsed:.2f} seconds\n")
#     return result

def determine_most_likely_cluster(prompt: str, goal: str) -> ClusterPrediction:
    """
    Uses LLM to analyze the user prompt and clustered map to determine
    the most likely cluster that contains the goal object.
    """
    print("3. .......Determining most likely cluster with LLM.........\n")
    
    # Format cluster information for the LLM
    cluster_descriptions = get_labeled_cluster_descriptions()
    
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
            
            response_json = extract_json_from_response(response_text, expected_keys=["reasoning"])
            if "location_confidence" not in response_json:
                response_json["location_confidence"] = 0.0

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

            # Clamp to [0.0, 1.0] for robustness
            try:
                response_json["location_confidence"] = float(response_json["location_confidence"])
            except (TypeError, ValueError):
                response_json["location_confidence"] = 0.0
            response_json["location_confidence"] = max(0.0, min(1.0, response_json["location_confidence"]))

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

    print(f"Selected Cluster: {result.cluster_id if result.cluster_id is not None else 'NONE'}")
    print(f"Reasoning: {result.reasoning}")
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

def _find_object_cluster(object_id: str) -> int | None:
    """Return cluster id for a given object id from clustered_map."""
    for entry in clustered_map:
        if entry.get("id") == object_id:
            try:
                return int(entry.get("cluster"))
            except (TypeError, ValueError):
                return None
    return None

def _build_goal_candidates(goal_objects: List[Dict]) -> List[Dict]:
    """Build compact candidate objects for final LLM goal selection."""
    candidates = []
    for obj in goal_objects:
        obj_id = obj.get("id")
        if not obj_id:
            continue

        pose = obj.get("pose_map") if isinstance(obj.get("pose_map"), dict) else {}
        candidates.append({
            "object_id": obj_id,
            "class": obj.get("name", ""),
            "similarity": float(obj.get("similarity_score", 0.0)),
            "cluster": _find_object_cluster(obj_id),
            "coords": {
                "x": float(pose.get("x", 0.0)),
                "y": float(pose.get("y", 0.0)),
                "z": float(pose.get("z", 0.0)),
            }
        })
    return candidates

def select_final_goal(prompt: str, goal: str, goal_objects: List[Dict], cluster_info: Dict | None) -> GoalSelection:
    """
    Final LLM decision over extracted objects.
    The model can prioritize location or similarity and can return no selection.
    """
    print("6. .......Selecting final goal object with LLM.........\n")

    candidates = _build_goal_candidates(goal_objects)
    if not candidates:
        return GoalSelection(
            selected_object_id="",
            decision_basis="none",
            reasoning="No candidate objects were found in the map."
        )

    cluster_context = {
        "cluster_id": cluster_info.get("cluster_id") if cluster_info else None,
        "location_confidence": float(cluster_info.get("location_confidence", 0.0)) if cluster_info else 0.0,
        "prioritize_location": bool(cluster_info.get("prioritize_location", False)) if cluster_info else False,
        "reasoning": cluster_info.get("reasoning", "") if cluster_info else ""
    }

    SYSTEM_PROMPT = load_prompt('select_goal.txt')
    user_payload = {
        "user_prompt": prompt,
        "goal_class": goal,
        "cluster_context": cluster_context,
        "candidates": candidates
    }

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)}
    ]

    start_time = time.time()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"   Retry attempt {attempt + 1}/{max_retries}")

            response_text = call_llm(msgs, temperature=0.2)
            response_json = extract_json_from_response(
                response_text,
                expected_keys=["selected_object_id", "decision_basis", "reasoning"]
            )
            result = GoalSelection.model_validate(response_json)

            allowed_basis = {"location", "similarity", "mixed", "none"}
            if result.decision_basis not in allowed_basis:
                result.decision_basis = "mixed"

            valid_ids = {c["object_id"] for c in candidates}
            if result.selected_object_id and result.selected_object_id not in valid_ids:
                print(f"Warning: LLM returned unknown object id {result.selected_object_id}, returning no match")
                result.selected_object_id = ""
                result.decision_basis = "none"

            if not result.selected_object_id:
                result.decision_basis = "none"

            break
        except (ValueError, json.JSONDecodeError):
            if attempt < max_retries - 1:
                print("   JSON parsing failed, retrying...")
                time.sleep(1)
                continue
            print(f"ERROR: Failed to get valid JSON after {max_retries} attempts")
            print(f"Last response: {response_text}\n")
            raise
        except Exception as e:
            print(f"ERROR during final goal selection: {type(e).__name__}")
            print(f"Error message: {str(e)}\n")
            raise

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Selected object id: {result.selected_object_id or 'NONE'}")
    print(f"Decision basis: {result.decision_basis}")
    print(f"Reasoning: {result.reasoning}")
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
    
    # 2. Read the map and find objects of the goal class (ranked by CLIP similarity)
    goal_objects, text_embedding = find_goal_objects(effective_goal, goal.clip_prompts)

    # 3. Extract text embedding and similarity scores from ranked objects
    print("5. .......Processing object similarities.........\n")
    text_embedding_list = None
    object_similarities = []

    if text_embedding is not None:
        text_embedding_list = text_embedding.astype(float).tolist()
    
    if goal_objects:
        # Extract similarity scores from objects
        for obj in goal_objects:
            sim_score = obj.get("similarity_score", 0.0)
            object_similarities.append({
                "object_id": obj.get("id"),
                "similarity": sim_score
            })
        
        if object_similarities:
            print(f"Computed similarities for {len(object_similarities)} objects")
            print(f"Top match: {object_similarities[0]['object_id']} with {object_similarities[0]['similarity']:.2f}% similarity\n")
    else:
        print("No objects to compute similarities for.\n")

    # 4. Determine most likely cluster using LLM
    cluster_prediction = determine_most_likely_cluster(prompt, effective_goal)
    cluster_info = None
    if cluster_prediction.cluster_id is not None and cluster_prediction.cluster_id in cluster_summaries:
        cluster_coords = compute_cluster_coords(cluster_prediction.cluster_id)
        cluster_dimensions = get_cluster_dimensions(cluster_prediction.cluster_id)
        cluster_info = {
            "cluster_id": cluster_prediction.cluster_id,
            "cluster_label": cluster_labels.get(cluster_prediction.cluster_id, "unknown"),
            "objects": cluster_summaries[cluster_prediction.cluster_id],
            "reasoning": cluster_prediction.reasoning,
            "location_confidence": cluster_prediction.location_confidence,
            "prioritize_location": cluster_prediction.location_confidence >= 0.5,
            "coords": cluster_coords,
            "dimensions": cluster_dimensions
        }
    else:
        cluster_info = {
            "cluster_id": None,
            "cluster_label": "",
            "objects": [],
            "reasoning": cluster_prediction.reasoning,
            "location_confidence": cluster_prediction.location_confidence,
            "prioritize_location": False,
            "coords": None,
            "dimensions": None
        }
    
    # 5. Determine action plan
    action = extract_action(prompt)

    logic = decide_logic(prompt)

    # 6. Final object-level goal selection (can return no match)
    final_selection = select_final_goal(prompt, effective_goal, goal_objects, cluster_info)
    selected_goal = None
    if final_selection.selected_object_id:
        selected_goal = next(
            (obj for obj in goal_objects if obj.get("id") == final_selection.selected_object_id),
            None
        )

    # Preserve the extracted user-intent goal even if no mapped object is selected.
    final_goal_class = effective_goal

    return NavResult(
        goal=final_goal_class,
        #related_object=goal.related_object,
        goal_objects=goal_objects,
        clip_prompts=goal.clip_prompts,
        text_embedding=text_embedding_list,
        object_similarities=object_similarities,
        action=action.action,
        logic=logic.logic,
        cluster_info=cluster_info,
        selected_goal={
            "object_id": final_selection.selected_object_id,
            "decision_basis": final_selection.decision_basis,
            "reasoning": final_selection.reasoning,
            "coords": selected_goal.get("pose_map") if selected_goal else None,
            "similarity": float(selected_goal.get("similarity_score", 0.0)) if selected_goal else None,
            "cluster": _find_object_cluster(final_selection.selected_object_id) if selected_goal else None
        } if selected_goal else None
    )

# ----------------- RESULT SAVING ----------------- #

def _objects_with_coords(objects: List[Dict]) -> List[Dict]:
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
            "location_confidence": float(result.cluster_info.get("location_confidence", 0.0)),
            "prioritize_location": bool(result.cluster_info.get("prioritize_location", False)),
            "coords": result.cluster_info.get("coords"),
            "dimensions": result.cluster_info.get("dimensions")
        }

    payload = {
        "timestamp": time.time(),
        "prompt": prompt,
        "goal": result.goal,
        #"related_object": result.related_object,
        "clip_prompts": result.clip_prompts,
        # "text_embedding": result.text_embedding,
        "goal_objects": _objects_with_coords(result.goal_objects),
        "selected_goal": result.selected_goal,
        "object_similarities": result.object_similarities,
        "cluster_info": cluster_info,
        "action": result.action,
        "logic": result.logic,
        "valid": valid,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    print(f"Saved robot command to: {output_path}")

# -------------------- MAIN -------------------- #

def main():
    global house_map, clustered_map, cluster_summaries, clip_processor
    initialize_model()
    initialize_clip_processor()

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
