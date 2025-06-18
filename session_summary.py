from helpers import ask_ollama
from memory_engine import add_memory

def summarize_session(user_id, chat_history):
    # Convert to plain text for summarization
    full_chat = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    prompt = f"""You are an assistant tasked with summarizing a conversation.
Extract the key facts and preferences expressed by the user that would be useful to remember for future interactions.

Conversation:
full_chat

Summarize only factual or relevant details. Do not include small talk or filler.
"""

    summary = ask_ollama(prompt)
    if summary.strip():
        print("\n🧠 Summary generated and added to memory:")
        print(summary)
        add_memory(user_id, summary, memory_type="summary")
    else:
        print("No summary generated.")
