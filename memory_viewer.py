from memory_engine import collection

def list_user_memory(user_id, memory_type=None):
    filters = {"user": user_id}
    if memory_type:
        filters["type"] = memory_type

    results = collection.get(where=filters)
    docs = results.get("documents", [])
    ids = results.get("ids", [])
    metas = results.get("metadatas", [])

    if not docs:
        print("❌ No memory found.")
        return

    print(f"\n📚 Memory entries for '{user_id}' (type: {memory_type or 'all'}):\n")

    for i, (doc, meta, _id) in enumerate(zip(docs, metas, ids), 1):
        print(f"{i}. [{meta.get('type')}] {doc}")
        print(f"   ID: {_id}\n")

def main():
    print("🧠 Memory Viewer")
    user_id = input("Enter username: ").strip()
    mtype = input("Filter by memory type? (press enter to skip): ").strip().lower() or None

    list_user_memory(user_id, memory_type=mtype)

if __name__ == "__main__":
    main()
