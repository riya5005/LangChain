from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv() #load openai APIKey

llm = ChatOpenAI(model="gpt-5") #created object of OPENAI
result=llm.invoke("What is the capital of India")
print(result)
 