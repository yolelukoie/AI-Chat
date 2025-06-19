from ollama_api import ask_ollama
from memory_engine import add_memory

def extract_and_store_facts(user_id: str, conversation: str):
    prompt = f"""
You are a fact-extracting assistant.

Below is a conversation. Extract **only clear factual statements** about the user (preferences, interests, personal details, locations, facts).

Return the results as a bullet list. Skip generic or unclear statements.

Conversation:
{conversation}

Facts:
"""

    facts = ask_ollama(prompt)
    if facts.strip():
        print("\n📌 Extracted facts:")
        print(facts)
        for fact in facts.strip().split("\n"):
            if fact.strip().startswith("-"):
                add_memory(user_id, fact.strip("- ").strip(), memory_type="fact")
