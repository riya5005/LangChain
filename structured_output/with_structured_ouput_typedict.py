from langchain_ollama import ChatOllama
#Annotated lets you add extra information (description or rules) to a data type.
from typing import TypedDict,Annotated

# Load local Llama model
model = ChatOllama(model="llama3.2")

# Define the output schema
class Review(TypedDict):
    summary: Annotated[str,"A brief summary of the review"]
    sentiment: Annotated[str,"Return sentiment of the review either neg ,pos or neutral"]

# Create a structured output model
structured_model = model.with_structured_output(Review)


# Invoke the model
result = structured_model.invoke(
    f"""
    The phone has an amazing display and the battery easily lasts all day.
The camera is excellent in daylight, but the phone heats up while gaming.
Overall, it is a good value for the price.

    
    """
)

print(result)
print(result['summary'])
print(result['sentiment'])