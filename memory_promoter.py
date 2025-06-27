from memory_engine import retrieve_memory_by_type, add_memory, get_collections
from ollama_api import ask_ollama
from datetime import datetime, timedelta
from memory_ranker import rank_memories

def promote_summaries_to_facts(user_id):
    summaries = retrieve_memory_by_type(user_id, "recent sessions", top_k=20, memory_type="summary")
    reference_text = "user interests, personal events, emotional experiences, daily life"
    top_summaries = rank_memories(summaries, reference_text=reference_text, max_facts=5)

    summary_list = [item["content"] for item in top_summaries]
    joined_summaries = "\n\n".join([f"{i+1}. {s}" for i, s in enumerate(summary_list)])

    prompt = f"""
    The following items are memory summaries from previous conversations with a user:

    {joined_summaries}

    Please return a numbered list of items that qualify as **stable personal facts** — things that reflect the user's identity, long-term traits, or relationships.

    Only include the items that qualify as stable facts, rewriting them in a clean factual form. Reply as a numbered list.
    """

    response = ask_ollama(prompt)
    print("🤖 Promoted facts extracted:\n", response)

    # Extract numbered list items from response
    lines = response.strip().splitlines()
    fact_lines = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
    facts = [line.partition(".")[2].strip() for line in fact_lines if "." in line]

    for fact in facts:
        add_memory(user_id, fact, memory_type="fact")
        print(f"✅ Promoted to dynamic fact: {fact}")

def run_memory_promotion(user_id):
    promote_summaries_to_facts(user_id)

def compress_old_memory(user_id, max_age_days=7, memory_type="fact"):
    # Step 1: Fetch all matching memory entries
    memory_collection, _ = get_collections(user_id)
    results = memory_collection.get(where={"user": user_id})
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

Please summarize them into 3–5 compact bullet points, merging duplicate or overlapping facts.

Avoid redundancy, but retain all unique, relevant details.

Facts:
{combined}

Summary:
"""

    summary = ask_ollama(prompt)
    print("\n🧠 Compressed summary:")
    print(summary)

    # Step 3: Save summary and delete originals
    add_memory(user_id, summary, memory_type="compressed")

    print(f"🗑 Deleting {len(old_ids)} outdated memory entries...")
    memory_collection.delete(ids=old_ids)
