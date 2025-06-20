import json
import os
import re

PROFILE_FILE = "memory.json"

def load_static_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_static_profile(data):
    with open(PROFILE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_static_profile(user_id: str, key, value):
    import json, os

    PROFILE_FILE = "memory.json"
    # Handle removal instructions
    if isinstance(fact, tuple) and isinstance(fact[0], str) and fact[0].lower() == "remove":
        _, path = fact
        try:
            section, subsection, value = [p.strip() for p in path.split("->")]
            if section in user_profile:
                if subsection in user_profile[section]:
                    if isinstance(user_profile[section][subsection], list):
                        if value in user_profile[section][subsection]:
                            user_profile[section][subsection].remove(value)
                            print(f"🗑 Removed '{value}' from {section} -> {subsection}")
                    elif isinstance(user_profile[section][subsection], dict):
                        if value in user_profile[section][subsection]:
                            del user_profile[section][subsection][value]
                            print(f"🗑 Removed '{value}' from {section} -> {subsection}")
                    elif user_profile[section][subsection] == value:
                        del user_profile[section][subsection]
                        print(f"🗑 Removed entire {subsection} in {section}")
        except Exception as e:
            print(f"⚠️ Failed to parse removal: {e}")
        profile[user_id] = user_profile
        save_static_profile(profile)
        return

    profile = load_static_profile()
    user_profile = profile.get(user_id, {})

    key, value = fact

    # Ignore empty value
    if isinstance(value, str) and not value.strip():
        print(f"⚠️ Ignored empty value for '{key}'")
        return

    # Handle list-style keys
    list_keys = {"Hobbies", "Interests", "Languages", "Goals", "Skills"}
    if key in list_keys:
        existing = user_profile.get(key, [])
        if not isinstance(existing, list):
            existing = [existing]
        items = [v.strip().capitalize() for v in value.split(",")]
        for item in items:
            if item and item not in existing:
                existing.append(item)
        user_profile[key] = existing

    # Handle structured nested fields
    elif isinstance(value, dict):
        if key == "Family":
            family = user_profile.setdefault("Family", {})
            for subkey, name in value.items():
                if subkey.lower() == "partner":
                    family["Partner"] = name
                elif subkey.lower() in ("daughter", "son", "child", "children"):
                    family.setdefault("Children", []).append(name)
        elif key == "Pets":
            pets = user_profile.setdefault("Family", {}).setdefault("Pets", {})
            for animal_type, names in value.items():
                animals = pets.get(animal_type, [])
                for name in [n.strip() for n in names.split(",")]:
                    if name and name not in animals:
                        animals.append(name)
                pets[animal_type] = animals
        else:
            # Other nested section
            section = user_profile.setdefault(key, {})
            for subkey, val in value.items():
                section[subkey.strip()] = val.strip()

    # Handle simple Key: Value
    else:
        user_profile[key] = value.strip().capitalize()

    profile[user_id] = user_profile
    save_static_profile(profile)
    print(f"📌 Added to static profile: {key}: {value}")
