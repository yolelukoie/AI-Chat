from ollama_api import ask_ollama
from memory_engine import add_memory
from profile_updater import update_static_profile
import json
from profile_updater import load_static_profile


def normalize_and_merge_facts(user_id: str, new_facts_text: str) -> dict:

    existing_profile = load_static_profile().get(user_id, {})

    prompt = f"""
You're an assistant that merges new identity facts into an existing user profile.

Goals:
- Resolve duplicate fields like "Daughter" vs "Children", or Pets inside/outside Family.
- Preserve all meaningful facts, but avoid redundant or outdated keys.
- Return a clean nested JSON structure with keys like:
  Age, Location, Family, Pets, Hobbies, Interests, Personality, Education, Occupation, Health, Food Preferences, etc.
- All children go under Family -> Children.
- All pets go under Family -> Pets.
- Don't include "Name" or "Section" keys.
- If a field already exists in the profile and isn't contradicted, keep it.
- Output valid JSON only — no commentary.
- Conversation style - if mentioned, overwrite it.

User ID: {user_id}

Existing profile:
{json.dumps(existing_profile, indent=2)}

New facts to integrate:
{new_facts_text}

Cleaned, merged JSON:
"""
    response = ask_ollama(prompt)
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        print("⚠️ LLM returned invalid JSON:")
        print(response)
        return existing_profile  # fallback


def summarize_session(user_id, chat_history):
    # Convert to plain text for LLM
    full_chat = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    # 🧠 Step 1: Extract stable identity facts and store to memory.json
    extract_profile_facts_from_chat(full_chat, user_id)

    # 📝 Step 2: Summarize entire session for dynamic memory
    summary_prompt = f"""You are an assistant tasked with summarizing a conversation.

Extract the key facts and preferences expressed by the user that would be useful to remember for future interactions.
Summarize only relevant or meaningful parts of the conversation — avoid filler or small talk.
Note: This summary will be saved to memory for future reference, don't ititiate a new conversation.
Conversation:
{full_chat}


"""
    summary = ask_ollama(summary_prompt)

    if summary.strip():
        add_memory(user_id, summary, memory_type="summary")
        print("\n🧠 Summary saved to memory:")
        print(summary)
    else:
        print("No summary generated.")

def extract_profile_facts_from_chat(full_chat: str, user_id: str):
    existing_profile = load_static_profile().get(user_id, {})
    prompt = f"""
You are a memory assistant that extracts long-term identity-related facts about a user.

1. Your first task is to identify solid facts from the conversation and assign each to a meaningful category. If there's no such category in examples, make a new one.

Only include facts that are:
- Relevant to the current user's perosnality, preferences, relationships, or background
- Long-term or stable (not temporary emotions or small talk)

✅ Format:
• <Category>: <Value>
• <Category>:
    <Subkey>: <Value>

Examples:
• Age: 35
• Location: Tel Aviv
• Food Preferences: Hates peanut butter
• Family:
    Partner: Stav
    Daughter: Vedana
• Pets:
    Dogs: Arti, Yuston
• Coversation Style preferences: as if we were very close friends, add more jokes

❌ Do NOT include:
- User’s name (assumed to be known)
- Any uncategorized or vague observations
- facts about any other users

Each fact must be stored as a structured key-value pair, using the category as the key.
One per line. Skip uncertain or vague statements.

2. Merge new identity facts into an existing user profile.
- Resolve duplicate fields like "Daughter" vs "Children", or Pets inside/outside Family.
- If a field already exists in the profile and isn't contradicted, keep it.
- Conversation style - if mentioned, overwrite it.

3. If the user explicitly tells you to forget something, remove it from the profile.
return is as a new fact with the key "remove" and value in format "Section -> Subsection -> Value".

User ID: {user_id}

Existing profile:
{json.dumps(existing_profile, indent=2)}

Conversation:
{full_chat}

Facts:
"""

    raw_facts_text = ask_ollama(prompt)
    print("\n📌 Extracted profile facts from chat:")
    print(raw_facts_text)

    updated_facts = {}
    for line in raw_facts_text.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            updated_facts[key.strip()] = value.strip()

    if updated_facts:
        update_static_profile(user_id, updated_facts)  # Now handles batch updates