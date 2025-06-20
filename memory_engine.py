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
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

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


def retrieve_memory(user_id, query_text, top_k=3, memory_type="fact"):
    embedding = embed([query_text])[0]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where={
            "$and": [
                {"user": user_id},
                {"type": memory_type}
            ]
        },
        include=["documents", "distances"]
    )

    if not results["documents"] or not results["documents"][0]:
        return []  # Nothing found, return safely

    return [{"content": doc, "score": max(0.0, min(1.0, 1 - dist))}
            for doc, dist in zip(results["documents"][0], results["distances"][0])]


def list_user_memory(user_id: str):
    results = collection.get(where={"user": user_id})
    return results["documents"]
