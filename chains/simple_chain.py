from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load local Llama model
model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

# Prompt Template
prompt = PromptTemplate(
    template="Generate 5 facts about {topic}.",
    input_variables=["topic"]
)

# Output Parser
parser = StrOutputParser()

# Create Chain
chain = prompt | model | parser

# Invoke Chain
result = chain.invoke({"topic": "cricket"})

print(result)