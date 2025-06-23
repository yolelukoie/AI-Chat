from pathlib import Path
import json
from datetime import datetime

INSTRUCTION_FILE = str(Path(__file__).resolve().parent / "user_instructions.json")

def _load_data():
    if not Path(INSTRUCTION_FILE).exists():
        return {}
    with open(INSTRUCTION_FILE, "r") as f:
        return json.load(f)

def _save_data(data):
    with open(INSTRUCTION_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_instructions(user_id: str):
    data = _load_data()
    return data.get(user_id, [])

def add_instruction(user_id: str, instruction_text: str):
    data = _load_data()
    if user_id not in data:
        data[user_id] = []
    data[user_id].append({
        "text": instruction_text,
        "timestamp": datetime.now().isoformat()
    })
    _save_data(data)

def remove_instruction(user_id: str, exact_text: str):
    data = _load_data()
    if user_id not in data:
        return False
    original = len(data[user_id])
    data[user_id] = [i for i in data[user_id] if i["text"] != exact_text]
    _save_data(data)
    return len(data[user_id]) < original

def list_instructions(user_id: str):
    return get_instructions(user_id)
