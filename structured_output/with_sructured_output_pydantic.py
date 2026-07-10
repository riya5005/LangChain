from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Load local Llama model
model = ChatOllama(model="llama3.2")

# Define the output schema using Pydantic
class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(
        description="Return the sentiment of the review: Positive, Negative, or Neutral"
    )

# Create a structured output model
structured_model = model.with_structured_output(Review)

# Invoke the model
result = structured_model.invoke("""
The phone has an amazing display and the battery easily lasts all day.
The camera is excellent in daylight, but the phone heats up while gaming.
Overall, it is a good value for the price.
""")

print(result)
print(result.summary)
print(result.sentiment)