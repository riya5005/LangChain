# given a topic AI -> we have 2 llms and want 1. generate linkdin post 2. generate tweet and get both outputs.

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence


model=ChatOllama(
    model="llama3.2",
    temperature=0.7
)
prompt1=PromptTemplate(
    template="Generate a Linkdin Post about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)
parser=StrOutputParser()


parallel_chain=RunnableParallel({
    'linkdin':RunnableSequence(prompt1,model,parser),
    'tweet':RunnableSequence(prompt2,model,parser)


})
result=parallel_chain.invoke({'topic','AI'})
print(result)