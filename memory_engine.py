import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client and collection
client = chromadb.Client()
collection = client.get_or_create_collection("user_memory")

# Load embedding model
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def embed(texts):
    return embedding_model.encode(texts).tolist()

def add_memory(user_id: str, memory_text: str):
    doc_id = f"{user_id}_{hash(memory_text)}"
    collection.add(
        documents=[memory_text],
        ids=[doc_id],
        metadatas=[{"user": user_id}],
        embeddings=embed([memory_text])
    )

def retrieve_memory(user_id: str, query: str, top_k: int = 3) -> list:
    results = collection.query(
        query_embeddings=embed([query]),
        n_results=top_k,
        where={"user": user_id}
    )
    return results["documents"][0] if results["documents"] else []

def list_user_memory(user_id: str):
    results = collection.get(where={"user": user_id})
    return results["documents"]
