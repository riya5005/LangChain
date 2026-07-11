from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda,RunnablePassthrough
from pydantic import BaseModel,Field
from typing import Literal

model=ChatOllama(
    model="llama3.2",
    temperature=0.7
)

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal["positive","negative"]=Field(description="Sentiment of the feedback")

structured_model=model.with_structured_output(Feedback)

prompt1=PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative.\n{feedback}",
    input_variables=["feedback"]
)

classifier_chain=RunnableParallel(
    feedback=RunnablePassthrough(),
    classification=prompt1|structured_model
)

prompt2=PromptTemplate(
    template="Write an appropriate response to this positive feedback.\n{feedback}",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="Write an appropriate response to this negative feedback.\n{feedback}",
    input_variables=["feedback"]
)

branch_chain=RunnableBranch(
    (
        lambda x:x["classification"].sentiment=="positive",
        RunnableLambda(lambda x:{"feedback":x["feedback"]["feedback"]})
        |prompt2
        |model
        |parser
    ),
    (
        lambda x:x["classification"].sentiment=="negative",
        RunnableLambda(lambda x:{"feedback":x["feedback"]["feedback"]})
        |prompt3
        |model
        |parser
    ),
    RunnableLambda(lambda x:"Could not determine sentiment")
)

chain=classifier_chain|branch_chain

result=chain.invoke(
    {
        "feedback":"This is a wonderful smartphone. Battery backup is excellent and camera quality is amazing."
    }
)

print(result)