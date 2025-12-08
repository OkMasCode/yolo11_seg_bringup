import ollama
import json
import os
import time
import sys
from pydantic import BaseModel
from typing import List, Optional

# --- CONFIGURATION ---
CHAT_MODEL = "llama3.2:3b"
MAP_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/map.json"
OUTPUT_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json"
OLLAMA_HOST = "http://localhost:11435"

# List of valid object classes commonly found in a house
# These represent static objects in a typical household environment
VALID_OBJECT_CLASSES = [
    "sink",
    "refrigerator",
    "oven",
    "microwave",
    "toaster",
    "dishwasher",
    "dining table",
    "kitchen counter",
    "stove",
    "cabinet",
    "sofa",
    "couch",
    "tv",
    "television",
    "armchair",
    "coffee table",
    "bookshelf",
    "lamp",
    "window",
    "door",
    "bed",
    "nightstand",
    "dresser",
    "closet",
    "mirror",
    "toilet",
    "bathtub",
    "shower",
    "sink",
    "bathroom mirror",
    "towel rack",
    "chair",
    "table",
    "desk",
    "chair",
    "light",
    "painting",
    "picture frame",
    "plant",
    "potted plant",
    "rug",
    "carpet",
    "curtain",
    "blind",
    "wall",
    "floor",
    "ceiling",
    "bookshelf",
    "shelf",
    "box",
    "basket",
]

client = ollama.Client(host=OLLAMA_HOST)
house_map_objects = []  # List of objects from the map

# -------------------- CONNECTIVITY & MAP -------------------- #

def wait_for_server():
    """Ensures the Ollama container is reachable."""
    print(f"Connecting to Ollama brain at {OLLAMA_HOST}...")
    retries = 0
    while True:
        try:
            client.list()
            print("Successfully connected to Robot Brain.")
            return
        except Exception:
            retries += 1
            print(f"Waiting for container... (Attempt {retries})")
            time.sleep(2)
            if retries > 5:
                print("ERROR: Could not connect to Ollama.")
                sys.exit(1)

def load_house_map(filename):
    """
    Loads the map.json file containing all objects and their coordinates.
    
    Returns:
        list: List of objects with 'class' and 'coords' keys
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Map file not found: {filename}")
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data if isinstance(data, list) else []

# -------------------- TOOLS FOR LLM -------------------- #

def get_map_objects():
    """
    Tool: Returns all objects present in the house map with their classes.
    
    Returns:
        list: All unique object classes from the map
    """
    unique_classes = sorted(list(set(obj.get("class", "") for obj in house_map_objects if obj.get("class"))))
    return unique_classes

def check_object_in_map(object_name: str):
    """
    Tool: Check if a specific object exists in the house map.
    
    Args:
        object_name: The name of the object to search for
        
    Returns:
        dict: Contains 'found' (bool) and 'object' (dict if found)
    """
    object_name_lower = object_name.strip().lower()
    
    for obj in house_map_objects:
        if obj.get("class", "").lower() == object_name_lower:
            return {
                "found": True,
                "object": obj
            }
    
    return {
        "found": False,
        "object": None
    }

def find_semantically_related_object(goal_object: str):
    """
    Tool: Find an object in the map that is semantically related to the goal.
    
    This function returns ALL objects in the map so the LLM can reason about
    which one is most semantically related to the goal.
    
    Args:
        goal_object: The goal object name
        
    Returns:
        dict: Contains 'goal' and 'available_objects' list
    """
    return {
        "goal": goal_object,
        "available_objects": get_map_objects()
    }

# -------------------- MAIN LOGIC -------------------- #

class GoalExtraction(BaseModel):
    """Output from first LLM call: goal extraction and validation."""
    goal: str
    clip_prompt: str

class MapCheck(BaseModel):
    """Output from second LLM call: map verification."""
    goal: str
    goal_in_map: bool
    closest_object: Optional[str]

def extract_and_validate_goal(user_prompt: str):
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

VALID HOUSEHOLD OBJECTS (check synonyms carefully):
{valid_list_str}

Your task:
1. Extract the target object from the user's request, It MUST strictly map to one of these: [{valid_list_str}] (handle synonyms).
2. Extract 'features' formatted specifically for a CLIP Vision Model.
    - The feature MUST be a descriptive phrase including the object and its context/attributes.
    - DO NOT output isolated words like ["red", "small"].
    - DO output phrases like ["red bottle on the table"].
    - If no attributes/context are provided, just repeat the goal name.

Important: 
- Goal MUST be exactly one of the valid objects (or empty if no match)
- CLIP prompt should preserve colors, sizes, positions from original request, if no features are included do not add any
- If object not in valid list, return empty goal

Output JSON with: goal, clip_prompt"""

    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        format=GoalExtraction.model_json_schema()
    )
    
    try:
        result = GoalExtraction.model_validate_json(response["message"]["content"])
        return result
    except Exception as e:
        print(f"Error parsing first LLM call: {e}")
        return GoalExtraction(goal="", clip_prompt="")

def check_map_and_find_alternative(goal: str):
    """
    SECOND LLM CALL: Check if goal exists in map, find semantic alternative if not.
    
    Process:
    1. Use tools to check if the goal exists in the house map
    2. If goal is in map: return it
    3. If goal is NOT in map: use semantic reasoning to find closest related object
       Example: goal="toilet" not in map → closest="shower" (both bathroom fixtures)
    
    Args:
        goal: The validated goal from first LLM call
        
    Returns:
        MapCheck: Contains goal, whether it's in map, and closest alternative if needed
    """
    
    SYSTEM_PROMPT = f"""You are a semantic reasoning system for robot navigation.

Your task:
1. Use check_object_in_map tool to verify if the goal exists in the house map

2. If goal IS in the map:
   - Set goal_in_map=true
   - Set closest_object=null (not needed)

3. If goal is NOT in the map:
   - Use get_map_objects tool to see all available objects
   - Select the MOST semantically related object based on:
     * Same room type
     * Similar function or purpose
     * Physical proximity in typical house layout
   
   Examples of semantic relationships:
   - toilet → shower (both bathroom fixtures)
   - bed → nightstand (both bedroom furniture)
   - tv → sofa (both living room items)
   - sink → stove (both kitchen items)
   
   - Set goal_in_map=false
   - Set closest_object to the semantically related object you found

Rules:
- Output ONLY objects that are in the map (do not guess the object, just compare the goal with the list of objects that are in the map)
- Output JSON with: goal (same as input), goal_in_map (boolean), closest_object (string or null)"""

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Goal to check: {goal}"}
    ]
    
    # Call with tools
    response = client.chat(
        model=CHAT_MODEL,
        messages=msgs,
        tools=[check_object_in_map, get_map_objects]
    )
    
    # Process tool calls if any
    if response["message"].get("tool_calls"):
        msgs.append(response["message"])
        
        for tool_call in response["message"]["tool_calls"]:
            func_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            
            # Execute the tool
            if func_name == "check_object_in_map":
                result = check_object_in_map(**args)
            elif func_name == "get_map_objects":
                result = get_map_objects()
            else:
                result = {"error": "unknown tool"}
            
            # Add tool result to messages
            msgs.append({
                "role": "tool",
                "content": json.dumps(result)
            })
    
    # Final response with structured output
    final_response = client.chat(
        model=CHAT_MODEL,
        messages=msgs,
        format=MapCheck.model_json_schema()
    )
    
    try:
        result = MapCheck.model_validate_json(final_response["message"]["content"])
        return result
    except Exception as e:
        print(f"Error parsing second LLM call: {e}")
        return MapCheck(goal=goal, goal_in_map=False, closest_object=None)

class NavigationResult(BaseModel):
    """Combined output format."""
    goal: str
    goal_in_map: bool
    closest_object: Optional[str]
    clip_prompt: str

def process_navigation_request(user_prompt: str):
    """
    Two-stage LLM processing pipeline.
    
    Stage 1: Extract and validate goal from user prompt
    Stage 2: Check map and find semantic alternatives if needed
    
    Returns:
        NavigationResult: Complete navigation information
    """
    
    # FIRST LLM CALL: Extract goal and create CLIP prompt
    print("Stage 1: Extracting goal from prompt")
    goal_extraction = extract_and_validate_goal(user_prompt)
    
    if not goal_extraction.goal:
        print("Could not extract valid goal from prompt")
        return NavigationResult(
            goal="",
            goal_in_map=False,
            closest_object=None,
            clip_prompt=""
        )
    
    print(f"   Goal extracted: {goal_extraction.goal}")
    print(f"   CLIP prompt: {goal_extraction.clip_prompt}")
    
    # SECOND LLM CALL: Check map and find alternatives
    print("🗺️  Stage 2: Checking map for goal...")
    map_check = check_map_and_find_alternative(goal_extraction.goal)
    
    print(f"   Goal in map: {map_check.goal_in_map}")
    if map_check.closest_object:
        print(f"   Closest alternative: {map_check.closest_object}")
    
    # Combine results
    return NavigationResult(
        goal=goal_extraction.goal,
        goal_in_map=map_check.goal_in_map,
        closest_object=map_check.closest_object,
        clip_prompt=goal_extraction.clip_prompt
    )

# -------------------- FILE OUTPUT -------------------- #

def save_result(result: NavigationResult, original_prompt: str):
    """
    Saves the final navigation command to a JSON file.
    
    Output format:
    {
        "timestamp": <unix timestamp>,
        "prompt": <original user input>,
        "goal": <matched household object>,
        "closest_object": <object from map if goal not in map>,
        "clip_prompt": <CLIP search phrase>,
        "valid": <boolean - true if goal was successfully matched>
    }
    """
    
    # Get coordinates if closest_object is specified
    closest_coords = None
    if result.closest_object:
        obj_data = check_object_in_map(result.closest_object)
        if obj_data["found"]:
            closest_coords = obj_data["object"].get("coords")
    
    data = {
        "timestamp": time.time(),
        "prompt": original_prompt,
        "goal": result.goal if result.goal else None,
        "closest_object": result.closest_object,
        "closest_object_coords": closest_coords,
        "clip_prompt": result.clip_prompt,
        "valid": result.goal != ""
    }
    
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Command written to: {os.path.abspath(OUTPUT_FILE)}")
    except IOError as e:
        print(f"Error writing output file: {e}")

# -------------------- MAIN -------------------- #

def main():
    global house_map_objects
    
    wait_for_server()
    house_map_objects = load_house_map(MAP_FILE)
    map_objects = get_map_objects()
    
    print("House map loaded.")
    print(f"Objects in map: {map_objects}")
    print(f"Valid classes: {len(VALID_OBJECT_CLASSES)}")

    while True:
        try:
            user_prompt = input("\nRobot Goal -> ").strip()
            if not user_prompt: 
                continue
            if user_prompt.lower() in ["q", "quit", "exit"]: 
                break

            # Process the navigation request using LLM with tools
            result = process_navigation_request(user_prompt)
            
            # Save result to JSON file
            save_result(result, user_prompt)

            # Display output in the required format
            print("-" * 60)
            if result.goal:
                print(f"Goal: {result.goal}")
                if result.closest_object:
                    print(f"Closest Object: {result.closest_object}")
                else:
                    print(f"Closest Object: (goal is directly in map)")
            else:
                print(f"Goal: (could not determine)")
                print(f"Closest Object: N/A")
            
            print(f"CLIP Prompt: {result.clip_prompt if result.clip_prompt else 'N/A'}")
            print("-" * 60)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
