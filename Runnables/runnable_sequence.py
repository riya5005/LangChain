from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

model=ChatOllama(
    model="llama3.2",
    temperature=0.7
)
prompt=PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)
parser=StrOutputParser()

chain = RunnableSequence(prompt, model, parser)
result=chain.invoke({'topic':'AI'})

print(result)