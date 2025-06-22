from helpers import *
import threading

# to run a model: ollama run llama3.2
# or: ollama run deepseek-r1:7b, but also change the model in ollama_api.py
# others: py -m pip install
# 

def main():
    print("Ollama welcomes you!")
    print("You can chat with it, and it will try to respond based on your memory.\n")
    
    chat()

if __name__ == "__main__":
    main()

threading._DummyThread.__del__ = lambda self: None
