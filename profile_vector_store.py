import json
from datetime import datetime
from chromadb import PersistentClient
from memory_engine import embed
from pathlib import Path
from ollama_api import ask_ollama


PROFILE_FILE = str(Path(__file__).resolve().parent / "memory.json")
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
    embedding = embed(query)

    results = collection.query(
        query_embeddings=[embedding],
        where={"user": user_id},
        n_results=top_k
    )

    return results['documents'][0] if results['documents'] else []
