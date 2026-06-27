from langchain_openai import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-sonnet-4-20250514"
)
result=model.invoke("What is the capital of India")
print(result)

