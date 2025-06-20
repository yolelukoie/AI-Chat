from helpers import *

# command: ollama run llama3.2
# others: py -m install

def main():
    print("Welcome to the Ollama Chatbot!")
    print("This chatbot remembers your preferences and interests.")
    print("You can chat with it, and it will try to respond based on your memory.\n")
    
    chat()

if __name__ == "__main__":
    main()

    