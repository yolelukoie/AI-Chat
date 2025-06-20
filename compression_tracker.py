import json, os
from datetime import datetime, timedelta

COMPRESS_FILE = "last_compress.json"

def should_run_compression(user_id, days=3):
    if not os.path.exists(COMPRESS_FILE):
        return True
    with open(COMPRESS_FILE, "r") as f:
        data = json.load(f)
    last_run = data.get(user_id)
    if not last_run:
        return True
    last_time = datetime.fromisoformat(last_run)
    return datetime.now() - last_time > timedelta(days=days)

def update_compression_time(user_id):
    data = {}
    if os.path.exists(COMPRESS_FILE):
        with open(COMPRESS_FILE, "r") as f:
            data = json.load(f)
    data[user_id] = datetime.now().isoformat()
    with open(COMPRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)
