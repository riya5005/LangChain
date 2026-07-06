from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Pydantic Schema
class Person(BaseModel):
    name: str = Field(description="Name of the fictional person")
    age: int = Field(description="Age of the fictional person")
    city: str = Field(description="City where the fictional person lives")

# JSON Parser
parser = JsonOutputParser(pydantic_object=Person)

# Prompt
template = PromptTemplate(
    template="""
Give me the name, age and city of a fictional person.

{format_instruction}
""",
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

prompt = template.format()

print(prompt)

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
print(final_result["name"])
print(final_result["age"])
print(final_result["city"])