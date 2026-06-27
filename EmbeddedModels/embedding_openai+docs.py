from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding= OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)
documents=[
    "Delhi is the capital of India",
    "Kolkata is the capital of West bengal",
    "Riya is genius of World"
]
result=embedding.embed_documents(documents)
print(str(result))

#THIS CODE RUNS QUERY OF MULTIPLE EMBEDDING