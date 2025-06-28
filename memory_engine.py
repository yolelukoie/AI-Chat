import uuid
import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from datetime import datetime
from pathlib import Path
from ollama_api import ask_ollama

# Initialize ChromaDB client and collection
client = PersistentClient(path="chroma_store")

CHROMA_PATH = "chroma_store"
MEMORY_FACT = "fact"
MEMORY_SUMMARY = "summary"
MEMORY_COMPRESSED = "compressed"
print("🔄 memory_engine initialized, collections loaded.")

# Load embedding model
embedding_model = SentenceTransformer("D:/AI Chat/paraphrase-multilingual-MiniLM-L12-v2")

def get_collections(user_id: str):
    client = PersistentClient(path="chroma_store")
    memory_collection = client.get_or_create_collection(f"memory_{user_id}")
    profile_collection = client.get_or_create_collection(f"profile_{user_id}")
    return memory_collection, profile_collection

def embed(texts):
    return embedding_model.encode(texts).tolist()

# region General Memory Functions
def add_memory(user_id: str, memory_text: str, memory_type: str = "general", memory_collection = None):
    doc_id = f"{user_id}_{uuid.uuid5(uuid.NAMESPACE_DNS, memory_text + memory_type)}"
    if memory_collection is None:
        memory_collection, _ = get_collections(user_id)
    memory_collection.add(
        documents=[memory_text],
        ids=[doc_id],
        metadatas=[{
            "user": user_id,
            "type": memory_type,
            "timestamp": datetime.now().isoformat()
        }],
        embeddings=embed([memory_text])
    )

def retrieve_memory_by_type(user_id: str, query: str, top_k: int = 10, memory_collection = None) -> list:
    if memory_collection is None:
        memory_collection, _ = get_collections(user_id)
    results = memory_collection.query(
        query_texts=[query],
        n_results=top_k * 3,  # we'll filter after
        where={"user": user_id},
        include=["documents", "metadatas", "distances"]
    )

    # Zip results together and sort by distance (closer = more relevant)
    hits = []
    if results.get("documents") and results["documents"][0]:
        for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            hits.append({
                "content": doc,
                "type": meta.get("type", "general"),
                "score": 1 - score,  # convert distance to similarity
                "metadata": meta
            })

    return hits

def list_user_memory(user_id: str, memory_collection = None):
    if memory_collection is None:
        memory_collection, _ = get_collections(user_id)
    results = memory_collection.get(where={"user": user_id})
    return results["documents"] if results.get("documents") else []

def is_memory_removal_request(user_text: str) -> bool:
    prompt = f"""A user said: "{user_text}"

    Does the user clearly want to delete some memory or information stored about them?

    Only reply with YES or NO."""
    try:
        reply = ask_ollama(prompt).strip().upper()
        print("Is memory removal request?")
        return "YES" in reply
    except:
        return False

def find_and_remove_matching_memory(user_id: str, user_text: str, memory_collection = None, profile_collection = None):
    if memory_collection is None:
        memory_collection, profile_collection = get_collections(user_id)
    removed_any = False

    # 1. Embed the query
    embedding = embed([user_text])

    # 2. Search general memory
    mem_results = memory_collection.query(
        query_embeddings=[embedding],
        where={"user": user_id},
        n_results=1
    )
    if mem_results["ids"]:
        mem_id = mem_results["ids"][0][0]
        memory_collection.delete(ids=[mem_id])
        print(f"🗑 Removed general memory: {mem_results['documents'][0][0]}")
        removed_any = True

    # 3. Search profile memory
    prof_results = profile_collection.query(
        query_embeddings=[embedding],
        where={"user": user_id},
        n_results=1
    )
    if prof_results["ids"]:
        prof_id = prof_results["ids"][0][0]
        profile_collection.delete(ids=[prof_id])
        print(f"🗑 Removed profile vector: {prof_results['documents'][0][0]}")
        removed_any = True

    if not removed_any:
        print("⚠️ No matching memory found to delete.")
# endregion

# region Profile Memory Functions

def profile_to_description(user_profile: dict, user_id: str) -> str:
    prompt = f"""
You are a helpful assistant preparing a memory description for a user profile.

Below is structured profile data for a user named {user_id}. Take that data and convert it into a clear, readable summary that could help an assistant recall this person’s background, traits, preferences, and relationships.

❗Important:
- Do NOT invent memory.Don't invent anything — use only the information provided. Be concise and factual.

Structured profile data:
{json.dumps(user_profile, indent=2)}

Text summary:
"""
    print("Profile description:")
    return ask_ollama(prompt).strip()

def update_profile_vector(user_profile: dict, user_id: str, profile_collection = None):
    if profile_collection is None:
        _, profile_collection = get_collections(user_id)
    description = profile_to_description(user_profile, user_id)
    embedding = embed([description]) 

    profile_collection.upsert(
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

def query_profile_memory(user_id: str, query: str, top_k=3, profile_collection = None):
    if profile_collection is None:
        _, profile_collection = get_collections(user_id)
    if query == "__FULL__":
        results = profile_collection.get(where={"user": user_id})
        if results.get("documents"):
            print(f"📄 [__FULL__] Profile memory for '{user_id}':\n{json.dumps(results['documents'], indent=2)}")
            return "\n".join(results["documents"])
        return ""

    embedding = embed([query])
    results = profile_collection.query(query_embeddings=[embedding], where={"user": user_id}, n_results=top_k)
    if results.get("documents") and results["documents"][0]:
        print(f"📄 [Query: {query}] Profile search results for '{user_id}':\n{json.dumps(results['documents'], indent=2)}")
        return results["documents"][0]
    return []


def load_all_vector_profiles(user_id: str, profile_collection = None):
    if profile_collection is None:
        _, profile_collection = get_collections()
    # Get all user documents
    results = profile_collection.get(include=["documents", "metadatas"])
    
    profiles = {}
    for doc, meta in zip(results["documents"], results["metadatas"]):
        user = meta.get("user")
        if user:
            if user not in profiles:
                profiles[user] = []
            profiles[user].append(doc)

    # Combine each user's document chunks
    return {user: "\n".join(chunks) for user, chunks in profiles.items()}
# endregion