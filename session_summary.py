from ollama_api import ask_ollama
from memory_engine import add_memory
from profile_updater import update_static_profile

def summarize_session(user_id, chat_history):
    # Convert to plain text for LLM
    full_chat = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    # 🧠 Step 1: Extract stable identity facts and store to memory.json
    extract_profile_facts_from_chat(full_chat, user_id)

    # 📝 Step 2: Summarize entire session for dynamic memory
    summary_prompt = f"""You are an assistant tasked with summarizing a conversation.

Extract the key facts and preferences expressed by the user that would be useful to remember for future interactions.
Summarize only relevant or meaningful parts of the conversation — avoid filler or small talk.

Conversation:
{full_chat}

Note: This summary will be saved to memory for future reference.
"""
    summary = ask_ollama(summary_prompt)

    if summary.strip():
        add_memory(user_id, summary, memory_type="summary")
        print("\n🧠 Summary saved to memory:")
        print(summary)
    else:
        print("No summary generated.")

def extract_profile_facts_from_chat(full_chat: str, user_id: str):
    prompt = f"""
You are a memory assistant.

Your job is to extract long-term identity-related facts about the user based on the conversation below.
Return each fact in a strict structured format so it can be stored.

Use ONLY these formats:

✅ To add facts:
• Key: Value
• Section:
  Subkey: Value1, Value2

✅ To remove outdated or false facts:
• Remove: Section -> Subkey -> Value

Examples:

• Age: 35
• Location: Tel Aviv
• Hobbies: Meditation
• Family:
  Partner: Stav
  Daughter: Vedana
• Pets:
  Dogs: Arti, Yuston

To remove:
• Remove: Pets -> Dogs -> Yuston
• Remove: Family -> Partner -> Stav

Do NOT mention what is missing.
Only extract clear factual identity info or removal requests.

⚠️ Do not summarize the conversation.
⚠️ Only list clearly structured identity facts using "• Key: Value" or "• Section:\n  Subkey: Value"

Conversation:
{full_chat}

Facts (STRICT FORMAT ONLY — do not return prose or summaries):
"""

    prompt = f"User name: {user_id}\n\n" + prompt
    result = ask_ollama(prompt)
    print("\n📌 Extracted profile facts from chat:")
    print(result)

    lines = result.splitlines()
    facts = []
    current_section = None
    subitems = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.lower().startswith("• remove:"):
            try:
                path = stripped.split(":", 1)[1].strip()
                facts.append(("Remove", path))
            except:
                continue

        elif stripped.startswith("•"):
            if current_section:
                facts.append((current_section, subitems))
                current_section = None
                subitems = {}

            content = stripped.lstrip("•").strip()
            if ":" in content:
                key, value = content.split(":", 1)
                if value.strip() == "":
                    current_section = key.strip()
                    subitems = {}
                else:
                    facts.append((key.strip(), value.strip()))

        elif current_section and ":" in stripped:
            subkey, value = stripped.split(":", 1)
            subitems[subkey.strip()] = value.strip()

    if current_section:
        facts.append((current_section, subitems))

    for key, value in facts:
        update_static_profile(user_id, key, value)
