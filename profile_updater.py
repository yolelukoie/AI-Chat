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

def save_static_profile(data):
    with open(PROFILE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_static_profile(user_id: str, key, value):
    profile = load_static_profile()
    user_profile = profile.get(user_id, {})

    # Ignore reserved or unstructured keys
    if key.lower() in ("section", "name"):
        print(f"⚠️ Ignoring unstructured or reserved key: {key}")
        return

    # Ignore empty strings
    if isinstance(value, str) and not value.strip():
        print(f"⚠️ Ignored empty value for '{key}'")
        return

    # ✅ Handle explicit removals
    if isinstance(key, str) and key.lower() == "remove":
        try:
            section, subsection, val = [p.strip() for p in value.split("->")]
            if section in user_profile:
                if not isinstance(user_profile[section], dict):
                    print(f"⚠️ Cannot remove from non-dict section: {section}")
                    return
                if subsection in user_profile[section]:
                    target = user_profile[section][subsection]
                    if isinstance(target, list) and val in target:
                        target.remove(val)
                        print(f"🗑 Removed '{val}' from {section} -> {subsection}")
                    elif target == val:
                        del user_profile[section][subsection]
                        print(f"🗑 Removed exact match in {section} -> {subsection}")
        except Exception as e:
            print(f"⚠️ Failed to parse removal: {e}")
        profile[user_id] = user_profile
        save_static_profile(profile)
        update_profile_vector(user_profile, user_id)
        return

    # ✅ Handle nested dict facts (e.g., Family, Pets)
    if isinstance(value, dict):
        if key not in user_profile or not isinstance(user_profile[key], dict):
            user_profile[key] = {}  # Overwrite invalid or missing structure

        section = user_profile[key]

        for subkey, val in value.items():
            if isinstance(val, list):
                existing = section.get(subkey, [])
                if not isinstance(existing, list):
                    existing = [existing] if existing else []
                for item in val:
                    if item not in existing:
                        existing.append(item)
                section[subkey] = existing
            else:
                section[subkey] = val.strip() if isinstance(val, str) else val

    # ✅ Handle comma-separated strings as lists (e.g., Hobbies: Cooking, Yoga)
    elif isinstance(value, str) and "," in value:
        items = [v.strip().capitalize() for v in value.split(",")]
        existing = user_profile.get(key, [])
        if not isinstance(existing, list):
            existing = [existing] if existing else []
        for item in items:
            if item and item not in existing:
                existing.append(item)
        user_profile[key] = existing

    # ✅ Handle simple Key: Value
    else:
        user_profile[key] = value.strip().capitalize() if isinstance(value, str) else value

    profile[user_id] = user_profile
    save_static_profile(profile)
    update_profile_vector(user_profile, user_id)
    print(f"📌 Added to static profile: {key}: {value}")
