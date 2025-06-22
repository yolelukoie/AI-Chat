import requests
import json
from ollama_api import ask_ollama
from memory_engine import retrieve_memory, retrieve_memory_by_type, add_memory, client
from chat_history import load_history, save_history
from session_summary import summarize_session
from promotion_tracker import should_run_promotion, update_promotion_time
from memory_promoter import promote_summaries_to_facts as run_memory_promotion
from fact_extractor import extract_and_store_facts
from memory_promoter import compress_old_memory
from compression_tracker import should_run_compression, update_compression_time
from pathlib import Path
from profile_updater import load_static_profile
from profile_vector_store import profile_to_description


PROFILE_FILE = str(Path(__file__).resolve().parent / "memory.json")

# ----- region Memory Management -----
def load_all_memory(filename=PROFILE_FILE):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_user_memory(user_id, filename="PROFILE_FILE"):
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
    all_profiles = load_static_profile()
    print(f"\n🤖 Welcome, {memory.get('user_name', user_id)}! Type 'exit' to quit.\n")

    chat_history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("\n💬 Chat session ended. Summarizing...")
            try:
                summarize_session(user_id, chat_history)
            except Exception as e:
                print(f"⚠️ Failed to summarize session: {type(e).__name__}: {e}")
            if should_run_compression(user_id):
                compress_old_memory(user_id)
                update_compression_time(user_id)
            if should_run_promotion(user_id):
                run_memory_promotion(user_id)
                update_promotion_time(user_id)
            break

        if user_input.lower() == "/promote":
            run_memory_promotion(user_id)
            continue
        elif user_input.lower() == "/reflect":
            facts = retrieve_memory(user_id, "who is the user", top_k=10, memory_type="fact")
            print("\n🧠 Here's what I know about you:\n" if facts else "🤷 I don't know anything about you yet.\n")
            for f in facts:
                print(f"• {f['content']} (score: {round(f['score'], 2)})")
            continue
        elif user_input.lower() == "/compress":
            compress_old_memory(user_id)
            continue

        
        # --- MEMORY RETRIEVAL (current user) ---
        all_hits = retrieve_memory_by_type(user_id, user_input, top_k=15)

        summary_hits = [m for m in all_hits if m["type"] == "summary"]
        fact_hits    = [m for m in all_hits if m["type"] == "fact"]
        compressed_hits = [m for m in all_hits if m["type"] == "compressed"]

        summary_section = "\n".join([f"[PREVIOUS SESSION] {m['content']}" for m in summary_hits])

        compressed_chunks = [m["content"] for m in compressed_hits if m.get("score", 0) > 0.75]
        fact_chunks = [m["content"] for m in fact_hits if m.get("score", 0) > 0.95]

        memory_chunks = []
        if compressed_chunks:
            memory_chunks.append("[SUMMARY MEMORY]")
            memory_chunks.extend([f"• {chunk}" for chunk in compressed_chunks])
        if fact_chunks:
            memory_chunks.append("[DETAILED MEMORY]")
            memory_chunks.extend([f"• {chunk}" for chunk in fact_chunks])

        retrieved_text = summary_section + "\n\n" + "\n".join(memory_chunks)
        use_static_profile = not memory_chunks and not summary_section


        # --- STATIC MEMORY LOADING ---
        all_profiles = load_static_profile()
        static_profile = all_profiles.get(user_id, {})
        user_profile_section = ""
        if use_static_profile:
            user_profile_section = "\n\n[USER PROFILE]\n" + profile_to_description(static_profile, user_id)

        # --- Conversation style ---
        conversation_style = static_profile.get("Conversation Style")
        style_instruction = f"\n\n✦ Use a {conversation_style.lower()} tone when replying." if conversation_style else ""

        # --- OTHER USERS DETECTED? ---
        mentioned_profiles = []
        input_lower = user_input.lower()
        for name, profile in all_profiles.items():
            if name.lower() != user_id.lower() and name.lower() in input_lower:
                mentioned_profiles.append(name)

        if mentioned_profiles:
            full_reply = ""
            for mentioned_user in mentioned_profiles:
                # 📎 Inject both static and dynamic memory for that user
                static = profile_to_description(all_profiles[mentioned_user], mentioned_user)
                all_hits = retrieve_memory(mentioned_user, user_input, top_k=15)
                summary_hits = [m for m in all_hits if m["type"] == "summary"]
                fact_hits = [m for m in all_hits if m["type"] == "fact"]

                dynamic = []
                if summary_hits:
                    dynamic.append("[SUMMARY MEMORY]")
                    dynamic.extend([f"• {s['content']}" for s in summary_hits])
                if fact_hits:
                    dynamic.append("[FACT MEMORY]")
                    dynamic.extend([f"• {f['content']}" for f in fact_hits])

                other_prompt = (
                    f"You were asked about user {mentioned_user}.\n Pronouns in the next sentences like 'him' or 'her' are likely referring to this user. Use this information to answer accurately. Be concise, honest, and don’t invent anything."
                    f"\n[STATIC PROFILE]\n{static}"
                    f"\n\n[DYNAMIC MEMORY]\n{'\n'.join(dynamic) if dynamic else '(none found)'}"
                    f"\n\nUse these to personalize your answers.{style_instruction}"
                    f"\n\nUser: {user_input}"
                )

                print(f"📎 Injecting profile for: {mentioned_user}")
                reply = ask_ollama(other_prompt)
                print(f"Lama: {reply}")
                chat_history.append({"role": "assistant", "content": reply})
                save_history(user_id, chat_history)
                print(f"Lama: {reply}\n")
            continue  # Don't continue to general prompt below
        

        # --- FINAL COMBINED PROMPT ---
        instructions = """
        You are a helpful assistant with memory.

        The memory sections below include:
        - USER PROFILE: Facts about the current user
        - CONTEXTUAL MEMORY: Summarized and factual info from previous sessions

        Use these to personalize your answers.
        Reply in this conversation style: {style_instruction}.
        If a user or name appears in memory, respond as if you remember them.

        ❗Important:
        - Do NOT invent memory. Be honest about what you know and don’t know.
        """

        full_prompt = (
            f"User name: {user_id}\n\n"
            f"{user_profile_section}"
            f"\n\n{instructions.strip()}"
            f"\n\n[DEBUG MEMORY: If you reference memory, say exactly what you're referencing.]"
            f"\n\n[CONTEXTUAL MEMORY]\n{retrieved_text.strip()}"
            f"\n\nUser: {user_input}"
        )

        reply = ask_ollama(full_prompt)
        print(f"Lama: {reply}")
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": reply})
        save_history(user_id, chat_history)

