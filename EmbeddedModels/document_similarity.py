from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)
documents = [
    "Delhi is the capital of India and is famous for the Red Fort and India Gate.",
    "Python is a popular programming language used in web development, AI, and data science.",
    "The Himalayas are the highest mountain range in the world and are located in Asia.",
    "Football is one of the most popular sports and is played by millions of people worldwide.",
    "Artificial Intelligence enables machines to learn from data and make intelligent decisions."
]
query="tell me about Python"

doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)
scores=cosine_similarity([query_embedding],doc_embeddings)[0]
index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(documents[index])
print(score)
