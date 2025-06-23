import requests
import json
from ollama_api import ask_ollama
from memory_engine import retrieve_memory, retrieve_memory_by_type, add_memory, client
from fact_extractor import extract_and_store_facts
from pathlib import Path
from profile_updater import load_static_profile
from profile_vector_store import profile_to_description
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from instructions_store import get_instructions

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global in-memory cache
_memory_cache = {}

# ----- region Memory Management -----


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

def fuzzy_cache_retrieve(user_id: str, user_input: str, top_k=15, similarity_threshold=0.93):
    """
    Retrieve memory hits for a given user input, using semantic caching based on cosine similarity.
    """
    user_cache = _memory_cache.setdefault(user_id, {
        "last_input": None,
        "last_embedding": None,
        "last_hits": None
    })

    current_embedding = embedding_model.encode(user_input, convert_to_tensor=True)

    if user_cache["last_embedding"] is not None:
        similarity = float(util.pytorch_cos_sim(current_embedding, user_cache["last_embedding"])[0])
        if similarity > similarity_threshold:
            print(f"[CACHE] Reusing memory for user '{user_id}' (similarity = {similarity:.3f})")
            return user_cache["last_hits"]

    # If no match, fetch and cache
    new_hits = retrieve_memory_by_type(user_id, user_input, top_k=top_k)
    user_cache["last_input"] = user_input
    user_cache["last_embedding"] = current_embedding
    user_cache["last_hits"] = new_hits
    return new_hits


def filter_relevant(chunks, threshold=0.9):
    seen = set()
    result = []
    for m in chunks:
        score = m.get("score", 0)
        content = m["content"].strip()
        if score < threshold:
            continue
        # Remove near-duplicates
        key = content.lower().replace(" ", "")[:80]
        if key in seen:
            continue
        seen.add(key)
        result.append(content)
    return result

def chat(user_id: str, user_input: str, memory, all_profiles, instructions) -> str:
    # vector_profile = query_profile_memory(user_id, "who is the user")

    similarity_threshold = 0.93
    user_cache = {}

    # --- MEMORY RETRIEVAL (current user) ---
    all_hits = fuzzy_cache_retrieve(user_id, user_input, top_k=15)
    summary_hits = [m for m in all_hits if m["type"] == "summary"]
    fact_hits    = [m for m in all_hits if m["type"] == "fact"]
    compressed_hits = [m for m in all_hits if m["type"] == "compressed"]

    summary_section = "\n".join([f"[PREVIOUS SESSION] {m['content']}" for m in summary_hits])
    compressed_chunks = filter_relevant(compressed_hits, threshold=0.85)
    fact_chunks = filter_relevant(fact_hits, threshold=0.95)

    memory_chunks = []
    if compressed_chunks:
        memory_chunks.append("[SUMMARY MEMORY]")
        memory_chunks.extend([f"• {chunk}" for chunk in compressed_chunks])
    if fact_chunks:
        memory_chunks.append("[DETAILED MEMORY]")
        memory_chunks.extend([f"• {chunk}" for chunk in fact_chunks])

    max_chunks = 100
    all_chunks = compressed_chunks + fact_chunks
    all_chunks = all_chunks[:max_chunks]

    retrieved_text = summary_section + "\n\n" + "\n".join(memory_chunks)
    use_static_profile = not memory_chunks and not summary_section

    # --- STATIC MEMORY LOADING ---
    static_profile = all_profiles.get(user_id, {})
    user_profile_section = ""
    if use_static_profile:
        user_profile_section = "\n\n[USER PROFILE]\n" + profile_to_description(static_profile, user_id)

    # --- FINAL COMBINED PROMPT ---
    prompt = """
        You are a helpful assistant with memory.

        The memory sections below include:
        - USER PROFILE: Facts about the current user
        - CONTEXTUAL MEMORY: Summarized and factual info from previous sessions

        Use these to personalize your answers.
        Reply in this conversation style: {instructions}.
        If a user or name appears in memory, respond as if you remember them.

        ❗Important:
        - Do NOT invent memory. Be honest about what you know and don’t know.
        """
    full_prompt = (
                f"User name: {user_id}\n\n"
                f"{user_profile_section}"
                f"\n\n{prompt.strip()}"
                f"\n\n[DEBUG MEMORY: If you reference memory, say exactly what you're referencing.]"
                f"\n\n[CONTEXTUAL MEMORY]\n{retrieved_text.strip()}"
                f"\n\nUser: {user_input}"
            )

    reply = ask_ollama(full_prompt)
    print(f"Lama: {reply}")
    return reply


def chat_about_users(user_id: str, user_input: str, mentioned_users: list[str], all_profiles: dict) -> str:

    for mentioned_user in mentioned_users:
        static = profile_to_description(all_profiles[mentioned_user], mentioned_user)
        hits = fuzzy_cache_retrieve(mentioned_user, user_input, top_k=15)
        summary_hits = [m for m in hits if m["type"] == "summary"]
        fact_hits = [m for m in hits if m["type"] == "fact"]

        dynamic = []
        if summary_hits:
            dynamic.append("[SUMMARY MEMORY]")
            dynamic.extend([f"• {s['content']}" for s in summary_hits])
        if fact_hits:
            dynamic.append("[FACT MEMORY]")
            dynamic.extend([f"• {f['content']}" for f in fact_hits])

        other_prompt = (
            f"You were asked about user {mentioned_user}.\nPronouns in the next sentences like 'him' or 'her' are likely referring to this user. Use this information to answer accurately."
            f"\n[STATIC PROFILE]\n{static}"
            f"\n\n[DYNAMIC MEMORY]\n{'\n'.join(dynamic) if dynamic else '(none found)'}"
            f"\n\nUse these to personalize your answers."
            f"\n\nUser: {user_input}"
        )

        reply = ask_ollama(other_prompt)
        print(f"Lama: {reply}")

    return reply  # Last reply returned for now
