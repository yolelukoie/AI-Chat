import json
import os

HISTORY_DIR = "chats"
os.makedirs(HISTORY_DIR, exist_ok=True)

def get_history_path(user_id):
    return os.path.join(HISTORY_DIR, f"history_{user_id}.json")

def load_history(user_id):
    path = get_history_path(user_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_history(user_id, history):
    path = get_history_path(user_id)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
