from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
"""
BaseModel is the base class used to create data models. It automatically:

Validates input data
Converts data to the correct type when possible
Raises errors if the data is invalid

#field

Field() is used to provide extra information about a field, such as:

Default values
Descriptions
Validation rules
Examples
Constraints (minimum, maximum, length, etc.)
"""


from pydantic import BaseModel,Field

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name:str=Field(description="Name of the Person")
    age:int=Field(gt=18,description="Age of the Person")
    city:str=Field(description="Name of the city the person belongs to")

parser=PydanticOutputParser(pydantic_object=Person)
template=PromptTemplate(
    template="Generate the name,age and city of a fictional {place} person\n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)    
prompt=template.invoke({'place':'India'})
result=model.invoke(prompt)
final_result=parser.parse(result.content)
print(final_result)
