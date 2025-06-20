from ollama_api import ask_ollama
from memory_engine import add_memory
from profile_updater import update_static_profile

def is_informative_fact(text):
    t = text.strip().lower()
    return (
        len(t) > 10 and
        not t.startswith("no clear factual") and
        not t.startswith("i can't assist") and
        not t.startswith("i don't have any") and
        t != "dogs"
    )

def extract_and_store_facts(user_id: str, conversation: str):
    prompt = f"""
You are a fact-extracting assistant.

Below is a conversation. Extract only clear factual statements about the user (preferences, interests, personal details, locations, facts).

Return ONLY the results as a bullet list, one per line, without any introductions or explanations.

Conversation:
{conversation}

Facts:
"""

    facts = ask_ollama(prompt)
    if facts.strip():
        print("\n📌 Extracted facts:")
        print(facts)

        lines = [line.strip("-• ").strip() for line in facts.splitlines()
                if line.strip().startswith(("•", "-"))]

        filtered = [fact for fact in lines if is_informative_fact(fact)]

        for fact in filtered:
            followup = f"""The following sentence is a fact about the user:

"{fact}"

Is this a stable personal trait or enduring personal fact that should be remembered in the user's profile (e.g., identity, location, pets, education, occupation, relationships)?

Reply with only 'yes' or 'no'."""
            decision = ask_ollama(followup).strip().lower()
            print(f"🤖 LLM routing decision: {decision}")

            if decision.startswith("yes"):
                update_static_profile(user_id, key, value)
            else:
                add_memory(user_id, fact, memory_type="fact")
                print(f"✅ Stored in dynamic memory: {fact}")
