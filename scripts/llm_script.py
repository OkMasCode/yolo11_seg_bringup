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
    print(f"[DEBUG] Tool 'get_map_objects' called")
    print(f"[DEBUG] Tool output: {unique_classes}")
    return unique_classes

def check_object_in_map(object_name: str):
    """
    Tool: Check if a specific object exists in the house map.
    
    Args:
        object_name: The name of the object to search for
        
    Returns:
        dict: Contains 'found' (bool) and 'object' (dict if found)
    """
    print(f"[DEBUG] Tool 'check_object_in_map' called with object_name='{object_name}'")
    object_name_lower = object_name.strip().lower()
    
    for obj in house_map_objects:
        if obj.get("class", "").lower() == object_name_lower:
            result = {
                "found": True,
                "object": obj
            }
            print(f"[DEBUG] Tool output: {result}")
            return result
    
    result = {
        "found": False,
        "object": None
    }
    print(f"[DEBUG] Tool output: {result}")
    return result

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
    print(f"[DEBUG] Tool 'find_semantically_related_object' called with goal_object='{goal_object}'")
    result = {
        "goal": goal_object,
        "available_objects": get_map_objects()
    }
    print(f"[DEBUG] Tool output: {result}")
    return result

# -------------------- MAIN LOGIC -------------------- #

class GoalExtraction(BaseModel):
    """Output from first LLM call: goal extraction and validation."""
    goal: str
    clip_prompt: str

class MapCheck(BaseModel):
    """Output from second LLM call: map verification."""
    goal: str
    goal_in_map: bool
    closest_objects: List[str]

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

CRITICAL RULES - YOU MUST FOLLOW EXACTLY:

STEP 1: CHECK IF GOAL EXISTS IN MAP
- Use check_object_in_map tool with the exact goal string
- This returns {{"found": true/false, "object": ...}}

STEP 2: DECIDE OUTPUT BASED ON RESULT

If goal IS FOUND in map (found=true):
   - Set goal_in_map=true
   - Set closest_objects to a list containing ONLY the goal: [goal]
   - Stop here, do not call get_map_objects

If goal is NOT FOUND in map (found=false):
   - Use get_map_objects tool to retrieve ALL available objects in the map
   - The tool returns a list of object class names that exist in the map
   - Select the 2 MOST semantically related objects from this list
   - Rank them by semantic similarity (most similar first)
   - Set goal_in_map=false
   - Set closest_objects to a list of exactly 2 objects: [most_similar, second_similar]

SEMANTIC MATCHING GUIDELINES:
Choose based on:
- Same room type (bathroom items together, kitchen items together)
- Similar function (seating → seating, cooking → cooking)
- Typical proximity in house layouts


ABSOLUTE REQUIREMENTS:
- If goal_in_map is true: closest_objects MUST be a list with one element: [goal]
- If goal_in_map is false: closest_objects MUST be a list of EXACTLY 2 strings from get_map_objects result
- NEVER invent object names not in the map
- If map has fewer than 2 objects, repeat objects to reach exactly 2 elements
- ALWAYS provide exactly 2 objects when goal is not in map

Output JSON with:
- goal: (exact same string as input)
- goal_in_map: (boolean)
- closest_objects: (list with [goal] if found in map, otherwise list of exactly 2 strings from map objects)"""

    print("[DEBUG] === SECOND LLM CALL: Map Check ===")
    print(f"[DEBUG] Goal to check: {goal}")
    
    # ALWAYS check if goal is in map first
    print("[DEBUG] Force-calling check_object_in_map tool")
    tool_start = time.time()
    map_check_result = check_object_in_map(goal)
    tool_end = time.time()
    print(f"[TIMER] check_object_in_map took: {tool_end - tool_start:.3f}s")
    
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Goal to check: {goal}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "check_object_in_map",
                    "arguments": {"object_name": goal}
                }
            }]
        },
        {
            "role": "tool",
            "content": json.dumps(map_check_result)
        }
    ]
    
    # If not found in map, let LLM call get_map_objects
    if not map_check_result["found"]:
        print("[DEBUG] Goal not in map, allowing LLM to call get_map_objects")
        llm_start = time.time()
        response = client.chat(
            model=CHAT_MODEL,
            messages=msgs,
            tools=[get_map_objects]
        )
        llm_end = time.time()
        
        print(f"[DEBUG] Raw LLM response (after map check): {response['message']}")
        print(f"[TIMER] LLM tool decision took: {llm_end - llm_start:.3f}s")
        
        # Process additional tool calls if any
        if response["message"].get("tool_calls"):
            print(f"[DEBUG] LLM requested {len(response['message']['tool_calls'])} additional tool call(s)")
            msgs.append(response["message"])
            
            for i, tool_call in enumerate(response["message"]["tool_calls"], 1):
                func_name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                print(f"[DEBUG] Tool call {i}: {func_name} with args: {args}")
                
                # Execute the tool
                if func_name == "get_map_objects":
                    result = get_map_objects()
                else:
                    result = {"error": "unknown tool"}
                    print(f"[DEBUG] Unknown tool requested: {func_name}")
                
                # Add tool result to messages
                msgs.append({
                    "role": "tool",
                    "content": json.dumps(result)
                })
                print(f"[DEBUG] Tool result added to conversation")
        else:
            print("[DEBUG] No additional tool calls requested by LLM")
    else:
        print("[DEBUG] Goal found in map, skipping get_map_objects call")
    
    # Final response with structured output
    print("[DEBUG] Requesting final structured output from LLM")
    final_start = time.time()
    final_response = client.chat(
        model=CHAT_MODEL,
        messages=msgs,
        format=MapCheck.model_json_schema()
    )
    final_end = time.time()
    
    print(f"[DEBUG] Raw LLM final response: {final_response['message']['content']}")
    print(f"[TIMER] Final structured output took: {final_end - final_start:.3f}s")
    
    try:
        result = MapCheck.model_validate_json(final_response["message"]["content"])
        print(f"[DEBUG] Parsed result: goal='{result.goal}', goal_in_map={result.goal_in_map}, closest_objects={result.closest_objects}")
        return result
    except Exception as e:
        print(f"[DEBUG] Error parsing second LLM call: {e}")
        return MapCheck(goal=goal, goal_in_map=False, closest_objects=[goal, goal, goal])

class NavigationResult(BaseModel):
    """Combined output format."""
    goal: str
    goal_in_map: bool
    closest_objects: List[str]
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
    print("\n" + "="*60)
    print("Stage 1: Extracting goal from prompt")
    print("="*60)
    stage1_start = time.time()
    goal_extraction = extract_and_validate_goal(user_prompt)
    stage1_end = time.time()
    
    if not goal_extraction.goal:
        print("[DEBUG] Could not extract valid goal from prompt")
        return NavigationResult(
            goal="",
            goal_in_map=False,
            closest_objects=[],
            clip_prompt=""
        )
    
    print(f"\n✓ Goal extracted: {goal_extraction.goal}")
    print(f"✓ CLIP prompt: {goal_extraction.clip_prompt}")
    print(f"[TIMER] ⏱️  Stage 1 total time: {stage1_end - stage1_start:.3f}s")
    
    # SECOND LLM CALL: Check map and find alternatives
    print("\n" + "="*60)
    print("Stage 2: Checking map for goal")
    print("="*60)
    stage2_start = time.time()
    map_check = check_map_and_find_alternative(goal_extraction.goal)
    stage2_end = time.time()
    
    print(f"\n✓ Goal in map: {map_check.goal_in_map}")
    if map_check.closest_objects:
        print(f"✓ Closest alternatives: {map_check.closest_objects}")
    print(f"[TIMER] ⏱️  Stage 2 total time: {stage2_end - stage2_start:.3f}s")
    
    total_time = stage1_end - stage1_start + stage2_end - stage2_start
    print(f"\n[TIMER] 🎯 TOTAL PROCESSING TIME: {total_time:.3f}s")
    
    # Combine results
    return NavigationResult(
        goal=goal_extraction.goal,
        goal_in_map=map_check.goal_in_map,
        closest_objects=map_check.closest_objects,
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
    
    # Get coordinates for all closest objects (only include objects that exist in map)
    closest_objects_with_coords = []
    verified_closest_objects = []
    if result.closest_objects:
        for obj_name in result.closest_objects:
            obj_data = check_object_in_map(obj_name)
            if obj_data["found"]:
                verified_closest_objects.append(obj_name)
                closest_objects_with_coords.append({
                    "class": obj_name,
                    "coords": obj_data["object"].get("coords")
                })
            else:
                print(f"[DEBUG] Warning: LLM returned object '{obj_name}' not in map, skipping")
    
    data = {
        "timestamp": time.time(),
        "prompt": original_prompt,
        "goal": result.goal if result.goal else None,
        "closest_objects": verified_closest_objects,
        "closest_objects_with_coords": closest_objects_with_coords,
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
            save_start = time.time()
            save_result(result, user_prompt)
            save_end = time.time()
            print(f"[TIMER] File save took: {save_end - save_start:.3f}s")

            # Display output in the required format
            print("-" * 60)
            if result.goal:
                print(f"Goal: {result.goal}")
                if result.goal_in_map:
                    print(f"Closest Objects: {result.closest_objects[0]}")
                    print(f"  (goal found directly in map)")
                else:
                    print(f"Closest Objects (Top 2 Alternatives):")
                    for i, obj in enumerate(result.closest_objects, 1):
                        print(f"  {i}. {obj}")
                    print(f"  (semantic alternatives - goal not in map)")
            else:
                print(f"Goal: (could not determine)")
                print(f"Closest Objects: N/A")
            
            print(f"CLIP Prompt: {result.clip_prompt if result.clip_prompt else 'N/A'}")
            print("-" * 60)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()