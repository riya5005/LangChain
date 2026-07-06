from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Define the response schema
schemas = [
    ResponseSchema(
        name="fact_1",
        description="Fact 1 about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="Fact 2 about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="Fact 3 about the topic"
    )
]

# Create the parser
parser = StructuredOutputParser.from_response_schemas(schemas)

# Prompt template
template = PromptTemplate(
    template="""
Give 3 facts about {topic}.

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Generate prompt
prompt = template.invoke({"topic": "Black Hole"})

# Invoke model
result = model.invoke(prompt)

# Parse output
final_result = parser.parse(result.content)

print(final_result)
print(final_result["fact_1"])
print(final_result["fact_2"])
print(final_result["fact_3"])