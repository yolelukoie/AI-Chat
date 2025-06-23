import requests
import re
import json

def ask_ollama(prompt, model="deepseek-r1:7b"):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": True
    }, stream=True)

    collected = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            collected += token

    print("[DEBUG] Raw response:", collected)
    return clean_deepseek_response(collected)
   
def clean_deepseek_response(text: str) -> str:
    # Remove <think>...</think> block
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()