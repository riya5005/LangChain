#given topic ->generating joke->print joke and total np. of words in the joke

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda

def word_count(text):
    return len(text.split())
model=ChatOllama(
    model="llama3.2",
    temperature=0.7
)
prompt=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)


parser=StrOutputParser()

joke_generator_chain=RunnableSequence(prompt,model,parser)


parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)


})
final_chain=RunnableSequence(joke_generator_chain,parallel_chain)

print(final_chain.invoke({'topic':'AI'}))
