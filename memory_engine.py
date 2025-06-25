from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from datetime import datetime
import uuid

# Initialize ChromaDB client and collection
client = PersistentClient(path="chroma_store")
collection = client.get_or_create_collection("user_memory")
print("🔄 memory_engine initialized, collection loaded.")
MEMORY_FACT = "fact"
MEMORY_SUMMARY = "summary"
MEMORY_COMPRESSED = "compressed"

# Load embedding model
embedding_model = SentenceTransformer("D:/AI Chat/paraphrase-multilingual-MiniLM-L12-v2")

def embed(texts):
    return embedding_model.encode(texts).tolist()

def add_memory(user_id: str, memory_text: str, memory_type: str = "general"):
    doc_id = f"{user_id}_{uuid.uuid5(uuid.NAMESPACE_DNS, memory_text + memory_type)}"
    collection.add(
        documents=[memory_text],
        ids=[doc_id],
        metadatas=[{
            "user": user_id,
            "type": memory_type,
            "timestamp": datetime.now().isoformat()
        }],
        embeddings=embed([memory_text])
    )

def retrieve_memory_by_type(user_id: str, query: str, top_k: int = 10) -> list:
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 3,  # we'll filter after
        where={"user": user_id},
        include=["documents", "metadatas", "distances"]
    )

    # Zip results together and sort by distance (closer = more relevant)
    hits = []
    for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({
            "content": doc,
            "type": meta.get("type", "general"),
            "score": 1 - score,  # convert distance to similarity
            "metadata": meta
        })

    return hits

def list_user_memory(user_id: str):
    results = collection.get(where={"user": user_id})
    return results["documents"]
