import requests
import re

def ask_ollama(prompt, model="deepseek-r1:7b"):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return clean_deepseek_response(response.json()["response"])
   
def clean_deepseek_response(text: str) -> str:
    # Remove <think>...</think> block
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()