import json, os
from datetime import datetime, timedelta

PROMOTION_FILE = "last_promote.json"

def should_run_promotion(user_id, days=1):
    if not os.path.exists(PROMOTION_FILE):
        return True

    with open(PROMOTION_FILE, "r") as f:
        data = json.load(f)

    last_run = data.get(user_id)
    if not last_run:
        return True

    last_time = datetime.fromisoformat(last_run)
    return datetime.now() - last_time > timedelta(days=days)

def update_promotion_time(user_id):
    data = {}
    if os.path.exists(PROMOTION_FILE):
        with open(PROMOTION_FILE, "r") as f:
            data = json.load(f)

    data[user_id] = datetime.now().isoformat()
    with open(PROMOTION_FILE, "w") as f:
        json.dump(data, f, indent=2)
