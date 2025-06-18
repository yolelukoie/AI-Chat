import requests
import json
from memory_engine import retrieve_memory, add_memory
from chat_history import load_history, save_history
from session_summary import summarize_session
from promotion_tracker import should_run_promotion, update_promotion_time
from memory_promoter import promote_summaries_to_facts


# ----- region Memory Management -----
def load_all_memory(filename="memory.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_user_memory(user_id, filename="memory.json"):
    all_memory = load_all_memory(filename)
    return all_memory.get(user_id, {})

def memory_to_prompt(memory):
    lines = []
    if "user_name" in memory:
        lines.append(f"The user's name is {memory['user_name']}.")
    if "language_preferences" in memory:
        langs = ", ".join(memory["language_preferences"])
        lines.append(f"The user speaks: {langs}.")
    if "interests" in memory:
        interests = ", ".join(memory["interests"])
        lines.append(f"The user is interested in: {interests}.")
    return "\n".join(lines)
# endregion

def ask_ollama(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

def chat():
    user_id = input("Enter your username: ").strip().lower()
    memory = load_user_memory(user_id)
    static_memory_prompt = memory_to_prompt(memory)

    print(f"\n🤖 Welcome, {memory.get('user_name', user_id)}! Type 'exit' to quit.\n")

    chat_history = load_history(user_id)

    while True:
        # Check for slash commands
        if user_input.lower() == "/promote":
            from memory_promoter import promote_summaries_to_facts
            promote_summaries_to_facts(user_id)
            continue

        elif user_input.lower() == "/reflect":
            facts = retrieve_memory(user_id, "who is the user", top_k=10, memory_type="fact")
            if facts:
                print("\n🧠 Reflection from memory:\n" + "\n".join(facts) + "\n")
            else:
                print("🤷 I don't know anything about you yet.\n")
            continue

        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("\n💬 Chat session ended. Summarizing...")
            try:
                summarize_session(user_id, chat_history)
            except Exception as e:
                print(f"⚠️ Failed to summarize session: {e}")

            break

        chat_history.append({"role": "user", "content": user_input})
        save_history(user_id, chat_history)

        # 🧠 Smart memory retrieval mode
        if "what do you know about me" in user_input.lower():
            memory_chunks = retrieve_memory(user_id, user_input, top_k=5, memory_type="fact")
        else:
            memory_chunks = retrieve_memory(user_id, user_input, top_k=3)

        retrieved_text = "\n".join(["[MEMORY] " + m for m in memory_chunks])

        # 🧠 Build full prompt
        prompt = f"{static_memory_prompt}\n\n{retrieved_text}\n\n" + \
                 "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history])

        # 💬 Get model response
        reply = ask_ollama(prompt)
        chat_history.append({"role": "assistant", "content": reply})
        save_history(user_id, chat_history)

        print(f"Ollama: {reply}\n")

        # 🧠 Optionally save the assistant’s new statements to memory
        if any(keyword in reply.lower() for keyword in ["you said", "you like", "your name", "you told me", "you live"]):
            add_memory(user_id, reply)