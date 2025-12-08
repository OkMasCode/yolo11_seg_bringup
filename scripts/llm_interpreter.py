import sys
import time
from unittest import result
import ollama
import json
import os
from pydantic import BaseModel
from typing import List

CHAT_MODEL = "llama3.2:3b"

OLLAMA_HOST = "http://localhost:11435"

MAP_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/map.json"

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

# ------------------ FUNCTIONS ----------------- #

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
    return sorted(list(set(obj.get("class", "") for obj in house_map if obj.get("class"))))

def find_objects(class_name: str):
    """Find all instances of an object in the map."""
    class_name_lower = class_name.strip().lower()
    matches = []
    
    for obj in house_map:
        if obj.get("class", "").lower() == class_name_lower:
            matches.append(obj)
    
    return matches

def find_goal_objects(goal: str):

    print("2. .......Extracting objects in the map.........\n")

    goal_objects = find_objects(goal)

    if goal_objects:
        print(f" {len(goal_objects)} objects of class {goal} found in the map")
        return goal_objects, []
    else:
        print(f"No objects of class {goal} found in the map.\n")
        start_time = time.time()
        alternatives_result = extract_alternatives(goal)
        end_time = time.time()
        elapsed = end_time - start_time
        print("Alternatives:")
        for obj in alternatives_result.alternatives:
            print(f"  • {obj}")
        print(f"Computation time: {elapsed:.2f} seconds\n")
        return [], alternatives_result.alternatives


# --------------- OUTPUT FORMATS --------------- #

class NavResult(BaseModel):
    goal: str # class of the object to navigate to
    goal_objects: List[str] # list of objects of that class in the map
    alternatives: List[str] # list of alternative classes if goal not found
    clip_prompt: str # prompt for CLIP model to find the object visually
    action: str # high-level action plan to reach the goal

class Goal(BaseModel):
    goal: str # class of the object to navigate to
    clip_prompt: str # prompt for CLIP model to find the object visually

class Alternatives(BaseModel):
    alternatives: List[str] # list of alternative classes if goal not found

class Action(BaseModel):
    action: str # Action that the robot has to perform

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

    Output JSON: {{"goal": "<exact string from list or empty>", "clip_prompt": "<string with features>"}}"""

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
    print(f"Clip Prompt: {result.clip_prompt}")
    print(f"Computation time: {elapsed:.2f} seconds\n")
    return result

def extract_alternatives(goal : str) -> Alternatives:

    map_objects = get_map_objects()

    print("2.2. .......Extracting alternatives from goal.........\n")
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

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Goal: {goal}\nAvailable objects in map: {map_objects}"}
    ]

    respose = client.chat(
        model = CHAT_MODEL,
        messages= msgs,
        format = Alternatives.model_json_schema()
    )
 
    result = Alternatives.model_validate_json(respose["message"]["content"])
    # Verify all alternatives exist in map (double check)
    alternatives = [a for a in result.alternatives if a in map_objects]
    alternatives = alternatives[:3]  # Ensure max 3
    return Alternatives(alternatives=alternatives)
 
def extract_action(prompt : str) -> Action:
        
    print("3. .......Extracting action from prompt.........\n")
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
        return NavResult(
            goal="",
            goal_objects=[],
            alternatives=[],
            clip_prompt="",
            action=""
        )
    # 2. Read the map and find objects of the goal class
    goal_objects, alternatives = find_goal_objects(goal.goal)

    # 3. Determine action plan
    action = extract_action(prompt) 

    return NavResult(
        goal=goal.goal,
        goal_objects=goal_objects,
        alternatives=alternatives,
        clip_prompt=goal.clip_prompt,
        action=action.action
    )

# -------------------- MAIN -------------------- #

def main():
    global house_map
    wait_for_server()

    house_map = load_house_map(MAP_FILE)
    map_objects = get_map_objects()

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
        elapsed = process_end - process_start
        # Final display
        print("="*60)
        print("RESULTS")
        print("="*60 + "\n")
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
        
        print(f"\nTotal processing time: {elapsed:.2f}s")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()