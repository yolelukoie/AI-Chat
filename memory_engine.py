import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from datetime import datetime


# Initialize ChromaDB client and collection
client = chromadb.Client()
collection = client.get_or_create_collection("user_memory")

# Load embedding model
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def embed(texts):
    return embedding_model.encode(texts).tolist()

def add_memory(user_id: str, memory_text: str, memory_type: str = "general"):
    doc_id = f"{user_id}_{hash(memory_text)}"
    collection.add(
        documents=[memory_text],
        ids=[doc_id],
        metadatas=[{
            "user": user_id,
            "type": memory_type,
            "created_at": datetime.now().isoformat()
        }],
        embeddings=embed([memory_text])
    )


def retrieve_memory(user_id: str, query: str, top_k: int = 3, memory_type: str = None) -> list:
    where_clause = {"user": user_id}
    if memory_type:
        where_clause = {
            "$and": [
                {"user": {"$eq": user_id}},
                {"type": {"$eq": memory_type}}
            ]
        }
    else:
        where_clause = {"user": {"$eq": user_id}}

    results = collection.query(
        query_embeddings=embed([query]),
        n_results=top_k,
        where=where_clause
    )

    return results["documents"][0] if results["documents"] else []


def list_user_memory(user_id: str):
    results = collection.get(where={"user": user_id})
    return results["documents"]
