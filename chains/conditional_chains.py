from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field
from typing import Literal

# Load Llama model
model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

parser = StrOutputParser()

# Pydantic schema
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )

# Structured Output Model
structured_model = model.with_structured_output(Feedback)

# Prompt for classification
prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback into either positive or negative.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
)

# Positive response prompt
prompt2 = PromptTemplate(
    template="""
A customer wrote the following positive feedback:

{feedback}

Write a short and polite thank-you response.
""",
    input_variables=["feedback"],
)

# Negative response prompt
prompt3 = PromptTemplate(
    template="""
A customer wrote the following negative feedback:

{feedback}

Write a short apology and assure the customer that the issue will be looked into.
""",
    input_variables=["feedback"],
)

# Classifier Chain
classifier_chain = RunnableParallel(
    feedback=RunnablePassthrough(),
    classification=prompt1 | structured_model
)

# Conditional Branch
branch_chain = RunnableBranch(
    (
        lambda x: x["classification"].sentiment == "positive",
        RunnableLambda(lambda x: {"feedback": x["feedback"]["feedback"]})
        | prompt2
        | model
        | parser,
    ),
    (
        lambda x: x["classification"].sentiment == "negative",
        RunnableLambda(lambda x: {"feedback": x["feedback"]["feedback"]})
        | prompt3
        | model
        | parser,
    ),
    RunnableLambda(lambda x: "Could not determine sentiment."),
)

# Final Chain
chain = classifier_chain | branch_chain

# Test
result = chain.invoke(
    {
        "feedback": "This is a wonderful smartphone. The battery backup is excellent and the camera quality is amazing."
    }
)

print(result)