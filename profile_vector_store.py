import json
from datetime import datetime
from chromadb import PersistentClient
from memory_engine import embed
from pathlib import Path
from ollama_api import ask_ollama

CHROMA_PATH = "chroma_store"
PROFILE_COLLECTION = "profile_vectors"

client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(PROFILE_COLLECTION)

def profile_to_description(user_profile: dict, user_id: str) -> str:
    prompt = f"""
You are a helpful assistant preparing a memory description for a user profile.

Below is structured profile data for a user named {user_id}. Your job is to convert it into a clear, readable summary that could help an assistant recall this person’s background, traits, preferences, and relationships.

Don't invent anything — use only the information provided. Be concise and factual.

Structured profile data:
{json.dumps(user_profile, indent=2)}

Text summary:
"""
    return ask_ollama(prompt).strip()

def update_profile_vector(user_profile: dict, user_id: str):
    description = profile_to_description(user_profile, user_id)
    embedding = embed(description)  # You use `embed`, not `embed_text`

    collection.upsert(
        documents=[description],
        embeddings=[embedding],
        metadatas=[{
            "user": user_id,
            "type": "profile",
            "timestamp": datetime.now().isoformat()
        }],
        ids=[f"profile_{user_id}"]
    )
    print(f"✅ Updated profile vector for {user_id}.")

def query_profile_memory(user_id: str, query: str, top_k=3):
    if query == "__FULL__":
        results = collection.query(where={"user": user_id}, n_results=100)
        return "\n".join(results["documents"][0]) if results["documents"] else ""

    embedding = embed(query)
    results = collection.query(query_embeddings=[embedding], where={"user": user_id}, n_results=top_k)
    return results['documents'][0] if results['documents'] else []

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

def load_all_vector_profiles():
    # Get all user documents
    results = collection.get(include=["documents", "metadatas"])
    
    profiles = {}
    for doc, meta in zip(results["documents"], results["metadatas"]):
        user = meta.get("user")
        if user:
            if user not in profiles:
                profiles[user] = []
            profiles[user].append(doc)

    # Combine each user's document chunks
    return {user: "\n".join(chunks) for user, chunks in profiles.items()}
