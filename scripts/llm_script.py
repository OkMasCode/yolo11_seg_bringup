import ollama
import json
import os
import time
import sys
from pydantic import BaseModel
from typing import List

# --- CONFIGURATION ---
CHAT_MODEL = "llama3.2:3b"
MAP_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/map.json"
OUTPUT_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json"
OLLAMA_HOST = "http://localhost:11435"

VALID_OBJECT_CLASSES = [
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
house_map_objects = []

# -------------------- CONNECTIVITY & MAP -------------------- #

def wait_for_server():
    """Ensures the Ollama container is reachable."""
    print(f"Connecting to Ollama brain at {OLLAMA_HOST}...")
    retries = 0
    while True:
        try:
            client.list()
            print("✓ Successfully connected to Robot Brain.\n")
            return
        except Exception:
            retries += 1
            print(f"Waiting for container... (Attempt {retries})")
            time.sleep(2)
            if retries > 5:
                print("ERROR: Could not connect to Ollama.")
                sys.exit(1)

def load_house_map(filename):
    """Loads the map.json file containing all objects and their coordinates."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Map file not found: {filename}")
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data if isinstance(data, list) else []

def get_map_objects():
    """Returns all unique object classes from the map."""
    return sorted(list(set(obj.get("class", "") for obj in house_map_objects if obj.get("class"))))

def find_objects_in_map(object_name: str):
    """Find all instances of an object in the map."""
    object_name_lower = object_name.strip().lower()
    matches = []
    
    for obj in house_map_objects:
        if obj.get("class", "").lower() == object_name_lower:
            matches.append(obj)
    
    return matches

# -------------------- TOOLS FOR LLM -------------------- #

class GoalExtraction(BaseModel):
    """Output from first LLM call: goal extraction and validation."""
    goal: str
    clip_prompt: str

class SemanticAlternatives(BaseModel):
    """Output from second LLM call: semantic alternatives."""
    alternatives: List[str]  # 3 semantically related objects

class ActionType(BaseModel):
    """Output from third LLM call: action determination."""
    action: str  # Either 'go_to_object' or 'bring_back_object'

class NavigationResult(BaseModel):
    """Combined output format."""
    goal: str
    goal_objects: List[dict]  # Full object data if found in map
    alternatives: List[str]  # Empty if goal found, otherwise 3 alternatives
    clip_prompt: str
    action: str

# -------------------- LLM CALLS -------------------- #

def extract_goal_from_prompt(user_prompt: str) -> GoalExtraction:
    """STEP 1: Extract goal and create CLIP prompt from user input."""
    print("→ Step 1: Extracting goal from prompt...")
    
    valid_list_str = ", ".join(VALID_OBJECT_CLASSES)
    
    SYSTEM_PROMPT = f"""You are a goal extraction system for a robot.

VALID HOUSEHOLD OBJECTS LIST:
{valid_list_str}

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

2. CLIP PROMPT CONSTRUCTION:
   - Format: "<features> <goal>" where goal is the exact object name
   - ALWAYS include the goal object name in the clip_prompt
   - If user provides features (color, size, position, etc.), include them BEFORE the goal
     * Example: "Go to red sofa" → clip_prompt: "red sofa"
     * Example: "Navigate to the small tv" → clip_prompt: "small television"
   - If NO features provided, clip_prompt should be ONLY the goal name
     * Example: "Go to sofa" → clip_prompt: "sofa"
   - DO NOT add features that were not mentioned by the user
   - DO NOT use isolated words, always include the object name

EXAMPLES:
Input: "Go to the black tv"
Output: {{"goal": "television", "clip_prompt": "black television"}}

Input: "Navigate to sofa"
Output: {{"goal": "sofa", "clip_prompt": "sofa"}}

Input: "Go to the red and large refrigerator"
Output: {{"goal": "refrigerator", "clip_prompt": "red large refrigerator"}}

Input: "Go to the car"
Output: {{"goal": "", "clip_prompt": ""}}

Output JSON: {{"goal": "<exact string from list or empty>", "clip_prompt": "<string with features>"}}"""

    start_time = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        format=GoalExtraction.model_json_schema()
    )
    elapsed = time.time() - start_time
    
    try:
        result = GoalExtraction.model_validate_json(response["message"]["content"])
        print(f"  ✓ Goal: '{result.goal}' | CLIP: '{result.clip_prompt}' ({elapsed:.2f}s)\n")
        return result
    except Exception as e:
        print(f"  ✗ Error parsing response: {e}\n")
        return GoalExtraction(goal="", clip_prompt="")

def get_semantic_alternatives(goal: str) -> SemanticAlternatives:
    """STEP 2B: Get semantic alternatives if goal not in map."""
    print("→ Step 2: Goal not in map, finding semantic alternatives...")
    
    map_objects = get_map_objects()
    
    SYSTEM_PROMPT = """You are a semantic reasoning system for robot navigation.

Your task: Given a GOAL object and a list of AVAILABLE objects in the house map, select the 3 MOST semantically related objects.

SEMANTIC MATCHING GUIDELINES:
1. Same room type (bathroom items together, kitchen items together, bedroom items together)
   - Bathroom: toilet, bathtub, shower, sink, towel rack, bathroom mirror
   - Kitchen: refrigerator, oven, microwave, stove, dishwasher, toaster, kitchen counter, cabinet
   - Bedroom: bed, nightstand, dresser, closet, mirror
   - Living room: sofa, tv, armchair, coffee table, lamp, bookshelf
   
2. Similar function (seating → seating, cooking → cooking, storage → storage)
   - Seating: sofa, couch, armchair, chair
   - Cooking/Food prep: oven, stove, microwave, toaster, kitchen counter
   - Storage: cabinet, dresser, closet, shelf, bookshelf
   - Lighting: lamp, light
   - Surfaces: table, dining table, coffee table, desk, kitchen counter

3. Typical proximity in house layouts
   - Adjacent rooms share borders
   - Common object pairings: bed+nightstand, sofa+coffee table, toilet+shower

SELECTION PROCESS:
1. Identify the category/function of the GOAL object
2. Find objects with the SAME CATEGORY from available objects
3. Rank by semantic similarity (most similar first)
4. Return EXACTLY 3 objects from the available list

CRITICAL RULES:
- Return EXACTLY 3 objects, ranked by similarity
- NEVER invent object names - only use objects from the available list
- NEVER add duplicates
- If fewer than 3 objects of same category, expand to other related categories
- Rank by semantic closeness (most similar first)

EXAMPLES:
Goal: "toilet" | Available: ["shower", "sink", "sofa", "bed"] → ["shower", "sink", "sofa"]
Goal: "bed" | Available: ["nightstand", "dresser", "sofa", "oven"] → ["nightstand", "dresser", "sofa"]
Goal: "tv" | Available: ["sofa", "armchair", "refrigerator"] → ["sofa", "armchair", "refrigerator"]

Output JSON: {"alternatives": ["object1_most_similar", "object2_second", "object3_third"]}"""

    start_time = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Goal: {goal}\nAvailable objects in map: {map_objects}"}
        ],
        format=SemanticAlternatives.model_json_schema()
    )
    elapsed = time.time() - start_time
    
    try:
        result = SemanticAlternatives.model_validate_json(response["message"]["content"])
        # Verify all alternatives exist in map
        alternatives = [a for a in result.alternatives if a in map_objects]
        alternatives = alternatives[:3]  # Ensure max 3
        
        print(f"  ✓ Alternatives: {alternatives} ({elapsed:.2f}s)\n")
        return SemanticAlternatives(alternatives=alternatives)
    except Exception as e:
        print(f"  ✗ Error parsing response: {e}\n")
        return SemanticAlternatives(alternatives=[])

def determine_action(user_prompt: str) -> ActionType:
    """STEP 3: Determine the robot action."""
    print("→ Step 3: Determining robot action...")
    
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

    start_time = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"User request: {user_prompt}"}
        ],
        format=ActionType.model_json_schema()
    )
    elapsed = time.time() - start_time
    
    try:
        result = ActionType.model_validate_json(response["message"]["content"])
        if result.action not in ["go_to_object", "bring_back_object"]:
            result.action = "go_to_object"
        
        print(f"  ✓ Action: '{result.action}' ({elapsed:.2f}s)\n")
        return result
    except Exception as e:
        print(f"  ✗ Error parsing response: {e}\n")
        return ActionType(action="go_to_object")
    """
    FIRST LLM CALL: Extract goal from prompt and validate against valid objects list.
    
    Process:
    1. Extract the goal object from user's natural language request
    2. Handle synonyms and map to valid object classes
       Example: "tv" in prompt → "television" (if that's in valid list)
    3. Generate CLIP prompt preserving original descriptive features
       Example: "Go to the black tv" → goal="television", clip_prompt="black tv"
    
    Args:
        user_prompt: User's natural language request
        
    Returns:
        GoalExtraction: Contains validated goal and clip prompt
    """
    
    valid_list_str = ", ".join(VALID_OBJECT_CLASSES)
    
    SYSTEM_PROMPT = f"""You are a goal extraction system for a robot.

VALID HOUSEHOLD OBJECTS LIST:
{valid_list_str}

CRITICAL RULES - YOU MUST FOLLOW EXACTLY:

1. GOAL EXTRACTION:
   - The 'goal' field MUST be EXACTLY one string from the valid objects list above
   - Match the EXACT string format from the list (case-sensitive)
   - Handle synonyms by mapping to the exact list entry:
     * "tv" → "television" (if "television" is in list)
     * "couch" → "sofa" (if "sofa" is in list)
   - If NO valid match exists, return goal as empty string ""
   - NEVER invent new object names
   - NEVER return objects not in the valid list

2. CLIP PROMPT CONSTRUCTION:
   - Format: "<features> <goal>" where goal is the exact object name
   - ALWAYS include the goal object name in the clip_prompt
   - If user provides features (color, size, position, etc.), include them BEFORE the goal
     * Example: "Go to red sofa" → clip_prompt: "red sofa"
     * Example: "Navigate to the small tv" → clip_prompt: "small television"
   - If NO features provided, clip_prompt should be ONLY the goal name
     * Example: "Go to sofa" → clip_prompt: "sofa"
   - DO NOT add features that were not mentioned by the user
   - DO NOT use isolated words, always include the object

EXAMPLES:
Input: "Go to the black tv"
Output: {{"goal": "television", "clip_prompt": "black television"}}

Input: "Navigate to sofa"
Output: {{"goal": "sofa", "clip_prompt": "sofa"}}

Input: "Go to the red and large refrigerator"
Output: {{"goal": "refrigerator", "clip_prompt": "red large refrigerator"}}

Input: "Go to the car"
Output: {{"goal": "", "clip_prompt": ""}}

Output JSON with: goal (exact string from list or empty), clip_prompt (always includes goal if valid)"""

    print("[DEBUG] === FIRST LLM CALL: Goal Extraction ===")
    print(f"[DEBUG] User prompt: {user_prompt}")
    
    start_time = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        format=GoalExtraction.model_json_schema()
    )
    end_time = time.time()
    
    print(f"[DEBUG] Raw LLM response: {response['message']['content']}")
    print(f"[TIMER] First LLM call took: {end_time - start_time:.3f}s")
    
    try:
        result = GoalExtraction.model_validate_json(response["message"]["content"])
        print(f"[DEBUG] Parsed result: goal='{result.goal}', clip_prompt='{result.clip_prompt}'")
        return result
    except Exception as e:
        print(f"[DEBUG] Error parsing first LLM call: {e}")
        return GoalExtraction(goal="", clip_prompt="")

def check_map_and_find_alternative(goal: str):
    """
    STEP 2: Check if goal exists in map, find semantic alternatives if not.
    """
    
    print("→ Step 2: Checking map for goal...")
    goal_objects = find_objects_in_map(goal)
    
    if goal_objects:
        print(f"  ✓ Found {len(goal_objects)} instance(s) of '{goal}' in map\n")
        return goal_objects, []
    else:
        print(f"  ✗ Goal not found in map\n")
        alternatives_result = get_semantic_alternatives(goal)
        return [], alternatives_result.alternatives

# -------------------- MAIN PIPELINE -------------------- #

def process_navigation_request(user_prompt: str) -> NavigationResult:
    """Main processing pipeline."""
    print("\n" + "="*60)
    print("PROCESSING REQUEST")
    print("="*60 + "\n")
    
    # STEP 1: Extract goal
    goal_result = extract_goal_from_prompt(user_prompt)
    
    if not goal_result.goal:
        print("✗ No valid goal extracted.\n")
        return NavigationResult(
            goal="",
            goal_objects=[],
            alternatives=[],
            clip_prompt="",
            action=""
        )
    
    # STEP 2: Check map
    goal_objects, alternatives = check_map_and_find_alternative(goal_result.goal)
    
    # STEP 3: Determine action
    action_result = determine_action(user_prompt)
    
    return NavigationResult(
        goal=goal_result.goal,
        goal_objects=goal_objects,
        alternatives=alternatives,
        clip_prompt=goal_result.clip_prompt,
        action=action_result.action
    )

# -------------------- FILE OUTPUT -------------------- #

def save_result(result: NavigationResult, original_prompt: str):
    """Save results to JSON file."""
    # Prepare goal objects with coordinates
    goal_objects_data = []
    for obj in result.goal_objects:
        goal_objects_data.append({
            "class": obj.get("class"),
            "coords": obj.get("coords")
        })
    
    # Prepare alternatives with coordinates
    alternatives_data = []
    for alt in result.alternatives:
        matches = find_objects_in_map(alt)
        if matches:
            alternatives_data.append({
                "class": alt,
                "coords": matches[0].get("coords")
            })
    
    data = {
        "timestamp": time.time(),
        "prompt": original_prompt,
        "goal": result.goal if result.goal else None,
        "goal_objects": goal_objects_data,
        "alternatives": alternatives_data,
        "clip_prompt": result.clip_prompt,
        "action": result.action,
        "valid": result.goal != ""
    }
    
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"✓ Results saved to: {os.path.abspath(OUTPUT_FILE)}\n")
    except IOError as e:
        print(f"✗ Error writing output file: {e}\n")

# -------------------- MAIN -------------------- #

def main():
    global house_map_objects
    
    wait_for_server()
    house_map_objects = load_house_map(MAP_FILE)
    map_objects = get_map_objects()
    
    print(f"📍 House map loaded with {len(house_map_objects)} objects")
    print(f"🏷️  Unique classes: {len(map_objects)}")
    print(f"📋 Valid classes recognized: {len(VALID_OBJECT_CLASSES)}\n")

    while True:
        try:
            user_prompt = input("Robot Goal → ").strip()
            if not user_prompt: 
                continue
            if user_prompt.lower() in ["q", "quit", "exit"]: 
                break

            overall_start = time.time()
            result = process_navigation_request(user_prompt)
            overall_time = time.time() - overall_start
            
            save_result(result, user_prompt)
            
            # Display results
            print("="*60)
            print("RESULT")
            print("="*60)
            if result.goal:
                print(f"Goal: {result.goal}")
                print(f"Action: {result.action}")
                print(f"CLIP Prompt: {result.clip_prompt}")
                
                if result.goal_objects:
                    print(f"\nFound in map ({len(result.goal_objects)} instance(s)):")
                    for obj in result.goal_objects:
                        print(f"  • {obj.get('class')} at {obj.get('coords')}")
                
                if result.alternatives:
                    print(f"\nSemantic alternatives (3 most related):")
                    for i, alt in enumerate(result.alternatives, 1):
                        print(f"  {i}. {alt}")
            else:
                print("Goal: (could not determine)")
                print("Action: N/A")
            
            print(f"\n⏱️  Total processing time: {overall_time:.2f}s")
            print("="*60 + "\n")

        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()