from langchain_ollama import ChatOllama

# Load local Llama model
model = ChatOllama(model="llama3.2")

# JSON Schema
review_schema = {
    "title": "Review",
    "description": "Schema for summarizing a smartphone review",
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "description": "Return the sentiment as Positive, Negative, or Neutral"
        }
    },
    "required": ["summary", "sentiment"]
}

# Create structured output model
structured_model = model.with_structured_output(review_schema)

# Invoke the model
result = structured_model.invoke("""
The phone has an amazing display and the battery easily lasts all day.
The camera is excellent in daylight, but the phone heats up while gaming.
Overall, it is a good value for the price.
""")

print(result)
print(result["summary"])
print(result["sentiment"])