# we will take a document and will generate 2 things i.e. Notes and 2. quiz and combine them and show user we can make this work parallely

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load local Llama model
model1 = ChatOllama(
    model="llama3.2",
    temperature=0.7
)
model2=ChatOllama(
    model="llama3.2",
    temperature=0.7
)
prompt1=PromptTemplate(
    template="Generate short and simple notes for the following text \n{text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="Generate 5 short question answers from the following text \n{text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="merge the provided notes and quiz into single document\n{notes} and {quiz}",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()
parallel_chain = RunnableParallel(
    notes=prompt1 | model1 | parser,
    quiz=prompt2 | model2 | parser
)

merge_chain=prompt3|model1|parser
chain=parallel_chain|merge_chain

text="""
**Cosmic Energy**

Cosmic energy is believed to be the universal life force that exists throughout the universe and connects all living and non-living things. According to many spiritual traditions, this energy flows through nature, the planets, stars, and every human being. It is often associated with peace, balance, and inner strength. People believe that practices such as meditation, yoga, and deep breathing help individuals connect with cosmic energy, improving mental clarity, emotional well-being, and physical health. While the concept of cosmic energy is widely accepted in spiritual and philosophical beliefs, it is not established as a measurable form of energy in mainstream scientific research. Regardless of one's perspective, the idea of cosmic energy encourages people to live in harmony with nature, maintain a positive mindset, and develop a deeper understanding of themselves and the universe.


"""

result=chain.invoke({'text':text})
print(result)