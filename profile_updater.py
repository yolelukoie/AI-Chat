import json
import os
import re
from profile_vector_store import update_profile_vector
from pathlib import Path


PROFILE_FILE = str(Path(__file__).resolve().parent / "memory.json")

def load_static_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_static_profile(user_id, profile: dict):
    with open("memory.json", "r") as f:
        all_data = json.load(f)

    all_data[user_id] = profile

    with open("memory.json", "w") as f:
        json.dump(all_data, f, indent=2)

def update_static_profile(user_id: str, facts_dict: dict):
    profile = load_static_profile()
    user_profile = profile.get(user_id, {})
    updated = False

    for key, value in facts_dict.items():
        # ✅ Handle explicit removals
        if isinstance(key, str) and key.lower() == "remove":
            try:
                section, subsection, val = [p.strip() for p in value.split("->")]
                if section in user_profile:
                    if not isinstance(user_profile[section], dict):
                        print(f"⚠️ Cannot remove from non-dict section: {section}")
                        continue
                    if subsection in user_profile[section]:
                        target = user_profile[section][subsection]
                        if isinstance(target, list) and val in target:
                            target.remove(val)
                            updated = True
                            print(f"🗑 Removed '{val}' from {section} -> {subsection}")
                        elif target == val:
                            del user_profile[section][subsection]
                            updated = True
                            print(f"🗑 Removed exact match in {section} -> {subsection}")
            except Exception as e:
                print(f"⚠️ Failed to parse removal: {e}")
            continue

        # ✅ Handle nested dicts (e.g., Family: {Partner: Stav})
        if isinstance(value, dict):
            if key not in user_profile or not isinstance(user_profile[key], dict):
                user_profile[key] = {}
            section = user_profile[key]
            for subkey, val in value.items():
                if isinstance(val, list):
                    existing = section.get(subkey, [])
                    if not isinstance(existing, list):
                        existing = [existing] if existing else []
                    for item in val:
                        if item not in existing:
                            existing.append(item)
                            updated = True
                    section[subkey] = existing
                else:
                    if section.get(subkey) != val:
                        section[subkey] = val.strip().capitalize() if isinstance(val, str) else val
                        updated = True

        # ✅ Handle comma-separated lists
        elif isinstance(value, str) and "," in value:
            items = [v.strip().capitalize() for v in value.split(",")]
            existing = user_profile.get(key, [])
            if not isinstance(existing, list):
                existing = [existing] if existing else []
            for item in items:
                if item and item not in existing:
                    existing.append(item)
                    updated = True
            user_profile[key] = existing

        # ✅ Handle simple key: value
        else:
            formatted_val = value.strip().capitalize() if isinstance(value, str) else value
            if user_profile.get(key) != formatted_val:
                user_profile[key] = formatted_val
                updated = True

        print(f"📌 Added/updated in static profile: {key}: {value}")

    if updated:
        profile[user_id] = user_profile
        save_static_profile(user_profile)
        update_profile_vector(user_profile, user_id)

