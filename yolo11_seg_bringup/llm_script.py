import ollama
import json
import os
import time
import sys
from pydantic import BaseModel
from typing import List

# --- CONFIGURATION ---
CHAT_MODEL = "llama3.2:3b"
MAP_FILE = "map.json"
# Explicitly point to the container's IP/Port
OLLAMA_HOST = "http://localhost:11435"

# Create the client instance to communicate with the container
client = ollama.Client(host=OLLAMA_HOST)

# Global map
house_map = {}

# -------------------- CONNECTIVITY CHECK -------------------- #

def wait_for_server():
    """Ensures the Ollama container is reachable before starting."""
    print(f"Connecting to Ollama brain at {OLLAMA_HOST}...")
    retries = 0
    while True:
        try:
            client.list() # Simple ping to check connection
            print("Successfully connected to Robot Brain (Container).")
            return
        except Exception:
            retries += 1
            print(f"Waiting for container... (Attempt {retries})")
            time.sleep(2)
            if retries > 5:
                print("ERROR: Could not connect to Ollama.")
                print("Ensure you ran: docker run ... --network host ...")
                sys.exit(1)

# -------------------- MAP LOADING & SUMMARY -------------------- #

def load_house_map(filename):
    """
    Load a structured house map from JSON.
    Expected format (keys are room names, values are lists of objects):
      {
        "kitchen": [
          {"object": "mug", "features": ["red"], "pos": [1.2, 0.5]},
          {"object": "plate", "features": [], "pos": [0.8, 0.4]}
        ],
        "bedroom": [
          {"object": "jacket", "features": ["black"], "pos": [3.1, -0.8]}
        ]
      }
    Everything is normalized to lowercase.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"House map file not found: {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = {}
    for room_name, objects in data.items():
        room_key = room_name.strip().lower()
        normalized_objects = []
        for obj in objects:
            name = str(obj.get("object", "")).strip().lower()
            features = [str(x).strip().lower() for x in obj.get("features", [])]
            pos = obj.get("pos", None)
            normalized_objects.append({
                "object": name,
                "features": features,
                "pos": pos,
            })
        normalized[room_key] = normalized_objects

    return normalized

def summarize_house_map(hmap):
    lines = []
    for room, objects in sorted(hmap.items()):
        names = sorted({obj.get("object", "") for obj in objects if obj.get("object")})
        lines.append(f"{room}: " + (", ".join(names) if names else "(empty)"))
    return "\n".join(lines)

# -------------------- TOOLS -------------------- #

def list_rooms():
    """
    Return a list of known rooms in the current house map.
    """
    return sorted(house_map.keys())

def objects_in_room(room):
    """
    Return all known objects in a given room.
    Args:
        room (str): room name (case-insensitive)
    """
    key = str(room).strip().lower()
    return house_map.get(key, [])

def rooms_with_object(object_name):
    """
    Return all rooms where an object with the given name appears.
    Args:
        object_name (str): e.g. "jacket"
    """
    name = str(object_name).strip().lower()
    found_rooms = []
    for room, objects in house_map.items():
        for obj in objects:
            if obj.get("object", "") == name:
                found_rooms.append(room)
                break
    return sorted(found_rooms)

# -------------------- MAIN LOGIC -------------------- #

class Rooms(BaseModel):
    Rooms: List[str]

class Goal(BaseModel):
    Goal: str
    Features: List[str]

def extract_room(user_prompt, house_map_summary):
    SYSTEM_PROMPT = f"""You are the language interface of a mobile robot in a house.

    You have TOOLS to query a house map:
    - list_rooms() -> list of room names
    - objects_in_room(room) -> objects known in that room
    - rooms_with_object(object_name) -> rooms where that object was seen

    Use these tools FIRST. Then use real-world knowledge about houses.

    Your tasks:

    1) Rank the rooms from the most likely to contain the target object to the least likely:
    - Answer EXACTLY following the json format:
            rooms: <list of room names, from most likely to least likely>

    Rules:
    - Prefer rooms where the object is in the map (use tools).
    - If the object is NOT in any room, do NOT give up.
    - Use list_rooms() + objects_in_room(room) to see which rooms contain
        RELATED objects, and use common sense.
    - Examples of related objects:
        * tshirt ~ jacket, trousers, wardrobe (clothes)
        * toilet paper ~ toilet, sink, bathroom cabinet
        * food item ~ fridge, stove, pan, pot, plates
    - Only use 'unknown' when you really cannot guess a room or goal.

    House map summary:
    {house_map_summary}
    """
    
    available_functions = {
        "list_rooms": list_rooms,
        "objects_in_room": objects_in_room,
        "rooms_with_object": rooms_with_object,
    }
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "User instruction (robot goal): " + user_prompt + "\n\n"
                "Remember: respond with:\n"
                "rooms: <list of room names, from most likely to least likely>"
            ),
        },    
    ]

    # --- Pass 1: Tool Decision ---
    t_start = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=messages,
        tools=[list_rooms, objects_in_room, rooms_with_object],
    )
    
    tool_calls = response["message"].get("tool_calls", [])

    if tool_calls:
        messages.append(response["message"]) # Keep conversation history
        
        for tool in tool_calls:
            fn_name = tool["function"]["name"]
            args = tool["function"]["arguments"]
            
            # Execute tool
            if fn_name in available_functions:
                result = available_functions[fn_name](**args)
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result), 
                })
                print(f"[TOOL] {fn_name}({args}) -> {result}")

    # --- Pass 2: Final Reasoning (Enforced JSON) ---
    final_response = client.chat(
        model=CHAT_MODEL,
        messages=messages,
        format=Rooms.model_json_schema(), # <--- Force JSON structure
    )
    t_end = time.time()
    print(f"[TIMING] Room Logic: {t_end - t_start:.3f}s")

    final_text = final_response["message"].get("content", "") or ""
    rooms = Rooms.model_validate_json(final_text)
    # build and return the Pydantic model
    return rooms

def extract_goal(user_prompt):
    SYSTEM_PROMPT = """You are a semantic parser for a mobile robot.
        From a user's instruction, extract:
        - goal: what the robot should bring, find, it is always a phisical object.
        This is usually:
            * a physical object (mug, tshirt, phone, book),
            * or a need that points to a type of object
            (e.g. "I am hungry" -> food / something to eat,
            "I am thirsty" -> something to drink).
        - feature: key attributes (color, owner, location phrase).
        Note that the Goal might not be explicitly mentioned in the instrunction, in 
        that case you need to infer it from context.

        Rules:
        - The goal is the THING the robot should address.
        - If the user says "bring me the tshirt that is on the table":
            goal: tshirt
            feature: on the table
        - If the user says "a glass of milk on the table":
            goal: glass of milk   (or just "milk in a glass")
            feature: on the table
        - Only choose them as goal if the user explicitly wants that thing as goal,
        e.g. "go to the toilet", "move the table".
        - If there is no explicit object (e.g. "I'm hungry"), set goal to a generic
        concept like "food" or "something to eat".
        - Add feature ONLY if they are specified in the instruction.
        - Do NOT infer the features.
        Output format (lowercase, exactly 2 lines):

        goal: <goal-string>
        feature: <comma-separated-features-or-none>

        No extra text."""

    t_start = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "User instruction: " + user_prompt,
            },
        ],
        format=Goal.model_json_schema(),
    )
    t_end = time.time()
    print(f"[TIMING] Goal Logic: {t_end - t_start:.3f}s")
    
    final_text = response["message"].get("content", "") or ""
    goal = Goal.model_validate_json(final_text)
    # build and return the Pydantic model
    return goal

def main():
    global house_map
    
    # 1. Health Check
    wait_for_server()

    # 2. Load Data
    house_map = load_house_map(MAP_FILE)
    house_map_summary = summarize_house_map(house_map)
    print("Map loaded.")

    while True:
        try:
            user_prompt = input("\nRobot Goal -> ").strip()
            if not user_prompt: continue
            if user_prompt.lower() in ["q", "quit", "exit"]: break

            # Run Logic
            rooms = extract_room(user_prompt, house_map_summary)
            print(rooms)

            best_room = rooms.Rooms[0]
            goal = extract_goal(user_prompt)
            goal_str = goal.Goal
            print(goal)

        except Exception as e:
            print(f"Error processing request: {e}")

if __name__ == "__main__":
    main()