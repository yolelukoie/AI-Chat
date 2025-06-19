from memory_engine import retrieve_memory, add_memory, collection
from ollama_api import ask_ollama
from datetime import datetime, timedelta
from memory_ranker import rank_memories


def promote_summaries_to_facts(user_id):
    summaries = retrieve_memory(user_id, "recent sessions", top_k=20, memory_type="summary")

    reference_text = "user interests, personal events, emotional experiences, daily life"
    top_summaries = rank_memories(summaries, reference_text=reference_text, max_facts=5)

    for item in top_summaries:
        content = item["content"]
        add_memory(user_id, content, memory_type="fact")
        print(f"✅ Promoted to fact: {content}")


def run_memory_promotion(user_id):
    promote_summaries_to_facts(user_id)

def compress_old_memory(user_id, max_age_days=7, memory_type="fact"):
    # Step 1: Fetch all matching memory entries
    results = collection.get(where={"user": user_id})
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    ids = results.get("ids", [])

    # Step 2: Filter by age
    now = datetime.now()
    cutoff = now - timedelta(days=max_age_days)

    old_entries = []
    old_ids = []

    for doc, meta, _id in zip(docs, metas, ids):
        created_at = meta.get("created_at")
        if meta.get("type") == memory_type and created_at:
            created_time = datetime.fromisoformat(created_at)
            if created_time < cutoff:
                old_entries.append(doc)
                old_ids.append(_id)

    if not old_entries:
        print("🧹 No old memory entries found for compression.")
        return

    combined = "\n".join(old_entries)
    prompt = f"""You are an assistant managing long-term memory.
Below are factual memory entries that are now more than {max_age_days} days old.

Please summarize them into 3–5 compact bullet points that retain all useful information.

Facts:
{combined}

Summary:
"""

    from ollama_api import ask_ollama
    summary = ask_ollama(prompt)

    print("\n🧠 Compressed summary:")
    print(summary)

    # Step 3: Save summary and delete originals
    add_memory(user_id, summary, memory_type="compressed")

    print(f"🗑 Deleting {len(old_ids)} outdated memory entries...")
    collection.delete(ids=old_ids)
