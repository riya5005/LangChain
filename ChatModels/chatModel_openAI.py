from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5")
result=model.invoke("What is the capital of India")
print(result)


#Output: (venv) PS C:\LangChain_Models> & C:\LangChain_Models\venv\Scripts\python.exe c:\LangChain_Models\LLMs\llm_demo.py

# content='The capital of India is New Delhi.' additional_kwargs={} response_metadata={
#     'token_usage': {
#         'completion_tokens': 9,
#         'prompt_tokens': 8,
#         'total_tokens': 17
#     },
#     'model_name': 'gpt-4',
#     'finish_reason': 'stop'
# } id='run-12345678-abcd-1234-efgh-1234567890ab'

# (venv) PS C:\LangChain_Models>