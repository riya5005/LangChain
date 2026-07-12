from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model=ChatOllama(
    model="llama3.2",
    temperature=0.7)

loader=PyPDFLoader('dogpdf.pdf')
docs=loader.load()
print(len(docs))

# prompt=PromptTemplate(
#     template="Write a summary for the following poem \n {poem}",
#     input_variables=['poem']
# )
