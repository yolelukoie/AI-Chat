import requests

system_prompt = (
    "You are a helpful assistant named Ollama. Only refer to memory if it's provided in the context block. "
    "If there is no relevant memory, do not say 'I don't remember'. Instead, acknowledge the input normally."
)

def ask_ollama(prompt, model="llama3.2"):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": system_prompt + "\n\n" + prompt,
        "stream": False
    })
    return response.json()["response"]
