from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

model = ChatOllama(model="llama3.2")

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Add user's message
    chat_history.append(HumanMessage(content=user_input))

    # Get AI response
    result = model.invoke(chat_history)

    # Add AI response
    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

print(chat_history)