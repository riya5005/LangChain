#you given a topic give to llm tell to generate detailed report then again tell to summarize in 5 lines
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt -> Detailed Report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

# 2nd Prompt -> Summarization
template2 = PromptTemplate(
    template="Write a 5-line summary of the following text:\n\n{text}",
    input_variables=["text"]
)

# Generate detailed report
prompt1 = template1.invoke({"topic": "Black Hole"})
result = model.invoke(prompt1)

# Generate summary
prompt2 = template2.invoke({"text": result.content})
result1 = model.invoke(prompt2)

print(result1.content)