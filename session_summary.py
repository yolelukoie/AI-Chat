from ollama_api import ask_ollama
from memory_engine import add_memory
from profile_updater import update_static_profile
import json
from profile_vector_store import update_profile_vector

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

from ollama_api import ask_ollama
from profile_vector_store import update_profile_vector
import json

def extract_profile_facts_from_chat(full_chat: str, user_id: str):
    prompt = (
        "You are an intelligent assistant extracting a user profile summary from a conversation.\n"
        "Read the full chat below and summarize everything that can be learned about the user in structured JSON format. \n"
        "Use information like location, profession, hobbies, interests, relationships, preferences, goals, etc.\n"
        "Only include information about the current user, not about other people whose name may sound in conversation.\n"
        "Do not invent any information, only include what is clearly stated or strongly implied.\n\n"
        f"CHAT:\n{full_chat}\n\n"
        "Return the output as a JSON dictionary. Do not explain anything."
    )

    response = ask_ollama(prompt)

    try:
        user_profile = json.loads(response)
        if isinstance(user_profile, dict):
            update_profile_vector(user_profile, user_id)
            print(f"✅ Profile vector updated for user {user_id}")
        else:
            print(f"⚠️ Unexpected profile format: {type(user_profile)}")
    except json.JSONDecodeError:
        print(f"❌ Failed to parse profile JSON from model response:\n{response}")
