def chatbot():
    print("Chatbot: Hello! How can I assist you?")
    while True:
        user_input = input("You: ").lower()
        if "hello" in user_input or "hi" in user_input:
            print("Chatbot: Hi there!")
        elif "your name" in user_input:
            print("Chatbot: I am your friendly chatbot.")
        elif "exit" in user_input:
            print("Chatbot: Goodbye!")
            break
        else:
            print("Chatbot: Sorry, I didn't understand that.")

chatbot()

import re

def chatbot():
    print("Chatbot: Hello! How can I assist you?")
    print("Chatbot: Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ").lower()

        # Greetings
        if re.search(r'\b(hello|hi|hey)\b', user_input):
            print("Chatbot: Hi there! How can I help you today?")
        
        # Asking about the chatbot
        elif re.search(r'\b(your name|who are you)\b', user_input):
            print("Chatbot: I am your friendly AI chatbot, here to assist you.")

        # Basic help queries
        elif re.search(r'\b(help|support|assist|issue|problem)\b', user_input):
            print("Chatbot: I'm here to help. Please describe your issue in detail.")
        
        # Asking about the time
        elif re.search(r'\b(time|current time)\b', user_input):
            from datetime import datetime
            print(f"Chatbot: The current time is {datetime.now().strftime('%H:%M:%S')}.")
        
        # Farewell
        elif re.search(r'\b(bye|goodbye|exit)\b', user_input):
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        # Default response
        else:
            print("Chatbot: Sorry, I didn't understand that. Can you rephrase?")

chatbot()
