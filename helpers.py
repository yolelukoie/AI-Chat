import requests
import json
from ollama_api import ask_ollama
from memory_engine import retrieve_memory, add_memory, client
from chat_history import load_history, save_history
from session_summary import summarize_session
from promotion_tracker import should_run_promotion, update_promotion_time
from memory_promoter import promote_summaries_to_facts as run_memory_promotion
from fact_extractor import extract_and_store_facts
from memory_promoter import compress_old_memory
from compression_tracker import should_run_compression, update_compression_time


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

def chat():
    user_id = input("Enter your username: ").strip()
    memory = load_user_memory(user_id)

    # Only include static profile memory if nothing else is available
    retrieved_memory_prompt = memory_to_prompt(memory)
    

    print(f"\n🤖 Welcome, {memory.get('user_name', user_id)}! Type 'exit' to quit.\n")

    chat_history = []

    while True:
        user_input = input("You: ")

        # Check for slash commands
        if user_input.lower() == "/promote":
            run_memory_promotion(user_id)
            continue

        elif user_input.lower() == "/reflect":
            facts = retrieve_memory(user_id, "who is the user", top_k=10, memory_type="fact")
            if facts:
                print("\n🧠 Here's what I know about you:\n")
                for f in facts:
                    print(f"• {f['content']} (score: {round(f['score'], 2)})")              
            else:
                print("🤷 I don't know anything about you yet.\n")
            continue

        elif user_input.lower() == "/compress":
            compress_old_memory(user_id)
            continue


        if user_input.lower() in ("exit", "quit"):
            print("\n💬 Chat session ended. Summarizing...")
            try:
                summarize_session(user_id, chat_history)
            except Exception as e:
                print(f"⚠️ Failed to summarize session: {type(e).__name__}: {e}")

            if should_run_compression(user_id):
                print("🧹 Compressing old memory...")
                compress_old_memory(user_id)
                update_compression_time(user_id)

            # 🧠 Check if we should promote summaries to facts
            if should_run_promotion(user_id):
                print("🔄 Running periodic memory promotion...")
                run_memory_promotion(user_id)
                update_promotion_time(user_id)

            break   

        chat_history.append({"role": "user", "content": user_input})
        save_history(user_id, chat_history)

        # 🧠 Retrieve most recent session summary (1 max)
        recent_summary = retrieve_memory(user_id, "previous conversation", top_k=5, memory_type="summary")
        summary_section = "\n".join([f"[PREVIOUS SESSION] {s}" for s in recent_summary]) if recent_summary else ""

        # 🔹 Retrieve and filter compressed memory
        compressed_raw = retrieve_memory(user_id, user_input, top_k=5, memory_type="compressed")
        compressed_chunks = [m["content"] for m in compressed_raw if m.get("score", 0) > 0.75]

        # 🔸 Retrieve and filter fact memory
        fact_raw = retrieve_memory(user_id, user_input, top_k=5, memory_type="fact")
        fact_chunks = [m["content"] for m in fact_raw if m.get("score", 0) > 0.75]

        # 🧠 Determine if we should use static profile memory
        use_static_profile = not (compressed_chunks or fact_chunks or recent_summary)
        user_profile_section = f"\n\n[USER PROFILE]\n{retrieved_memory_prompt.strip()}" if use_static_profile else ""

        # 🧠 Combine with labels
        memory_chunks = []
        if compressed_chunks:
            memory_chunks.append("[SUMMARY MEMORY]")
            memory_chunks.extend([f"• {chunk}" for chunk in compressed_chunks])
        if fact_chunks:
            memory_chunks.append("[DETAILED MEMORY]")
            memory_chunks.extend([f"• {chunk}" for chunk in fact_chunks])


        # 🧠 Combine all memory
        retrieved_text = summary_section + "\n\n" + "\n".join(memory_chunks)

        # 🧠 Build full prompt
        instructions_for_memory_use = """
        You are a helpful assistant with memory.

        The memory sections below include facts and summaries from past conversations with the user.
        Use them to personalize your replies.

        ❗Important:
        - Do NOT invent memory.
        - If a topic is not in memory and not mentioned in this conversation, say you don’t remember.
        - Be honest about what you know and don’t know.
        """
        full_prompt = (
            f"User name: {user_id}\n\n" +  # 👈 Add this line
            instructions_for_memory_use.strip() +
            "\n\n[DEBUG MEMORY: If you reference memory, say exactly what you're referencing.]" +
            user_profile_section +
            "\n\n[CONTEXTUAL MEMORY]\n" + retrieved_text.strip() +
            "\n\nUser: " + user_input.strip()
        )


        # 💬 Get model response
        reply = ask_ollama(full_prompt)
        chat_history.append({"role": "assistant", "content": reply})
        save_history(user_id, chat_history)

        print(f"Ollama: {reply}\n")