from memory_engine import retrieve_memory, add_memory, collection
from helpers import ask_ollama

def promote_summaries_to_facts(user_id):
    summaries = retrieve_memory(user_id, "recent sessions", top_k=20, memory_type="summary")

    if not summaries:
        print("No session summaries found.")
        return

    combined = "\n".join(summaries)

    prompt = f"""You're an assistant helping to manage long-term memory.
Below are multiple session summaries from a user named {user_id}.

Extract key **permanent facts** about the user's preferences, interests, identity, and important life details.
Only include information that is likely to be true across time.

Session summaries:
combined

Extract and return the facts as a bullet list.
"""

    result = ask_ollama(prompt)

    print("\n🧠 Promoted facts:")
    print(result)

    add_memory(user_id, result, memory_type="fact")

    # Optional: clean up old summaries
    # You could use collection.delete() with where={"user": user_id, "type": "summary"}
