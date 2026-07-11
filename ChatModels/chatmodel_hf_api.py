from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

import os

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-3B-Instruct",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HF_TOKEN")
# )
model = ChatOllama(model="llama3.2")
result = model.invoke("What is the capital of India?")
print(result.content)

