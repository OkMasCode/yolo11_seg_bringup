import ollama
import json
import os
import time
import sys
import difflib
from pydantic import BaseModel
from typing import List, Optional

# --- CONFIGURATION ---
CHAT_MODEL = "llama3.2:3b"
MAP_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/map.json"
OUTPUT_FILE = "/home/sensor/ros2_ws/src/yolo11_seg_bringup/config/robot_command.json"  # <--- File to store the result
OLLAMA_HOST = "http://localhost:11435"

# List of valid object classes from the provided input (COCO dataset labels)
VALID_OBJECT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]

client = ollama.Client(host=OLLAMA_HOST)
house_map = {}

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
    if not os.path.exists(filename):
        raise FileNotFoundError(f"House map file not found: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    normalized = {}
    for room, objs in data.items():
        r_key = room.strip().lower()
        n_objs = []
        for o in objs:
            n_objs.append({
                "object": str(o.get("object", "")).strip().lower(),
                "features": [str(x).strip().lower() for x in o.get("features", [])],
                "pos": o.get("pos", None)
            })
        normalized[r_key] = n_objs
    return normalized

def summarize_house_map(hmap):
    lines = []
    for room, objects in sorted(hmap.items()):
        names = sorted({obj.get("object", "") for obj in objects if obj.get("object")})
        lines.append(f"{room}: " + (", ".join(names) if names else "(empty)"))
    return "\n".join(lines)

# -------------------- TOOLS -------------------- #

def list_rooms(): return sorted(house_map.keys())
def objects_in_room(room): return house_map.get(str(room).strip().lower(), [])
def rooms_with_object(object_name):
    target = str(object_name).strip().lower()
    return sorted([r for r, objs in house_map.items() if any(o['object'] == target for o in objs)])

# -------------------- MAIN LOGIC -------------------- #

class Rooms(BaseModel):
    Rooms: List[str]

class Goal(BaseModel):
    Goal: str
    Features: List[str]

def validate_goal_object(raw_goal: str) -> Optional[str]:
    """Validates object existence in the closed-set class list."""
    cleaned = raw_goal.strip().lower()
    if cleaned in VALID_OBJECT_CLASSES: return cleaned
    matches = difflib.get_close_matches(cleaned, VALID_OBJECT_CLASSES, n=1, cutoff=0.6)
    return matches[0] if matches else None

def extract_room(user_prompt, house_map_summary):
    SYSTEM_PROMPT = f"""You are a robot navigation interface.
    1. Identify the target object.
    2. Use tools to find which room it is in.
    3. If unknown, infer the likely room based on common sense.
    
    House Summary: {house_map_summary}"""
    
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": f"Instr: {user_prompt}. JSON: rooms: [list]"}]
    
    resp = client.chat(model=CHAT_MODEL, messages=msgs, tools=[list_rooms, objects_in_room, rooms_with_object])
    if resp["message"].get("tool_calls"):
        msgs.append(resp["message"])
        for t in resp["message"]["tool_calls"]:
            fname = t["function"]["name"]
            args = t["function"]["arguments"]
            if fname == "list_rooms": res = list_rooms()
            elif fname == "objects_in_room": res = objects_in_room(**args)
            elif fname == "rooms_with_object": res = rooms_with_object(**args)
            else: res = "unknown tool"
            msgs.append({"role": "tool", "content": json.dumps(res)})
            
    final = client.chat(model=CHAT_MODEL, messages=msgs, format=Rooms.model_json_schema())
    return Rooms.model_validate_json(final["message"]["content"])

def extract_goal(user_prompt):
    """Extracts Goal and formats Features for CLIP."""
    valid_list_str = ", ".join(VALID_OBJECT_CLASSES)
    
    SYSTEM_PROMPT = f"""You are a semantic parser for a robot vision system.
    
    Tasks:
    1. Extract the 'goal' object. It MUST strictly map to one of these: [{valid_list_str}].
    
    2. Extract 'features' formatted specifically for a CLIP Vision Model.
       - The feature MUST be a descriptive phrase including the object and its context/attributes.
       - DO NOT output isolated words like ["red", "small"].
       - DO output phrases like ["red bottle on the table"].
       - If no attributes/context are provided, just repeat the goal name.
    
    Examples:
    - User: "Bring me the screwdriver" -> goal: "screwdriver", features: ["screwdriver"]
    - User: "Get the keys on the counter" -> goal: "keys", features: ["keys on the counter"]
    - User: "Find the red cup" -> goal: "cup", features: ["red cup"]
    
    Output JSON.
    """

    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format=Goal.model_json_schema(),
    )
    
    try:
        final_text = response["message"].get("content", "")
        goal_model = Goal.model_validate_json(final_text)
        
        # 1. Validate Object Class (Hard Restriction)
        validated_name = validate_goal_object(goal_model.Goal)
        if validated_name:
            goal_model.Goal = validated_name
        else:
            goal_model.Goal = "unknown"

        # 2. Validate/Fix CLIP Prompt
        if not goal_model.Features or goal_model.Features == [""]:
            goal_model.Features = [goal_model.Goal]
            
        return goal_model
        
    except Exception as e:
        print(f"Goal Parsing Error: {e}")
        return Goal(Goal="unknown", Features=["unknown"])

# -------------------- FILE OUTPUT -------------------- #

def save_result(rooms: Rooms, goal: Goal, original_prompt: str):
    """
    Saves the final navigation command to a JSON file.
    This file can be watched by a ROS2 node or C++ application.
    """
    data = {
        "timestamp": time.time(),
        "prompt": original_prompt,
        "valid": goal.Goal != "unknown",
        "navigation": {
            "target_rooms": rooms.Rooms,
            "priority_room": rooms.Rooms[0] if rooms.Rooms else None
        },
        "perception": {
            "target_class": goal.Goal,
            "clip_search_prompt": goal.Features[0]
        }
    }
    
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Command written to: {os.path.abspath(OUTPUT_FILE)}")
    except IOError as e:
        print(f"Error writing output file: {e}")

# -------------------- MAIN -------------------- #

def main():
    global house_map
    wait_for_server()
    house_map = load_house_map(MAP_FILE)
    house_map_summary = summarize_house_map(house_map)
    print("Map loaded.")
    print(f"Valid Classes: {VALID_OBJECT_CLASSES}")

    while True:
        try:
            user_prompt = input("\nRobot Goal -> ").strip()
            if not user_prompt: continue
            if user_prompt.lower() in ["q", "quit", "exit"]: break

            rooms = extract_room(user_prompt, house_map_summary)
            goal = extract_goal(user_prompt)

            # Save result to JSON file
            save_result(rooms, goal, user_prompt)

            print("-" * 30)
            if goal.Goal == "unknown":
                print("❌ Cannot physically detect this object class.")
            else:
                print(f"📍 Target Rooms:     {rooms.Rooms}")
                print(f"🎯 Detection Class:  '{goal.Goal}'")
                print(f"👁️ CLIP Prompt:      '{goal.Features[0]}'")
            print("-" * 30)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()