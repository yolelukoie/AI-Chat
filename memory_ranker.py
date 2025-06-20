from sentence_transformers import SentenceTransformer, util
import datetime

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight embedding model

def rank_memories(memories, reference_text, max_facts=5):
    """
    Rank memories based on similarity to reference text and recency.

    Args:
        memories (list of dicts): Each with 'content' and 'timestamp' (ISO format)
        reference_text (str): Summary of recent dialog, or user profile info
        max_facts (int): Max number of top items to return

    Returns:
        List of memory dicts (top-ranked)
    """
    if not memories:
        return []
    if len(memories) < 2:
        return memories  # No need to rank a single memory

    memory_texts = [m["content"] for m in memories]
    all_texts = memory_texts + [reference_text]
    embeddings = model.encode(all_texts, convert_to_tensor=True)

    memory_embeddings = embeddings[:-1]
    reference_embedding = embeddings[-1]
    similarities = util.cos_sim(memory_embeddings, reference_embedding).squeeze()


    now = datetime.datetime.utcnow()
    scored = []
    for i, memory in enumerate(memories):
        ts = memory.get("timestamp")
        if ts:
            try:
                age_minutes = (now - datetime.datetime.fromisoformat(ts)).total_seconds() / 60
                recency_boost = 1.0 / (1.0 + age_minutes / 60.0)
            except Exception:
                recency_boost = 1.0
        else:
            recency_boost = 1.0

        score = similarities[i].item() * recency_boost
        scored.append((score, memory))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_memories = [m for _, m in scored[:max_facts]]
    return top_memories
