from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model=ChatOllama(
    model="llama3.2",
    temperature=0.7)
prompt=PromptTemplate(
    template="Write a summary for the following poem \n {poem}",
    input_variables=['poem']
)

parser=StrOutputParser()

loader=TextLoader('ai.txt',encoding='utf-8')

docs=loader.load()
print(docs[0])

chain=prompt|model|parser

result=chain.invoke({'poem':docs[0].page_content})
print(result)