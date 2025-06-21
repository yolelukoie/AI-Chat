from helpers import *
import threading

# command: ollama run llama3.2
# others: py -m install

def main():
    print("Ollama welcomes you!")
    print("You can chat with it, and it will try to respond based on your memory.\n")
    
    chat()

if __name__ == "__main__":
    main()

threading._DummyThread.__del__ = lambda self: None
