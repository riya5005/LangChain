from pydantic import BaseModel,EmailStr,Field
#Optional means a field is not mandatory.

# If a field is marked as Optional, you can:

# provide a value, or
# leave it empty (it will usually become None if a default is given).
from typing import Optional

class Student(BaseModel):
    name:str
    #opetional filed
    age:Optional[int]=None
    #builtin validation
    email:EmailStr
    #field function
    cgpa:float=Field(gt=0,lt=10,description='A decimal value representing the cgpa of the student')

new_student={'name':"Riya","age":21,"email":"abc@gmail.com",'cgpa':5}

student=Student(**new_student)
student_dict=dict(student)

print(student)

