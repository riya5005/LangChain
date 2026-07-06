import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Load Llama model
model = ChatOllama(model="llama3.2")

# Prompt Template
template = PromptTemplate(
    input_variables=["genre", "character", "length"],
    template="""
Write a {length} {genre} story.

Main Character: {character}

The story should have:
- A title
- A beginning
- A happy or interesting ending

Use simple and easy English.
"""
)

# Streamlit UI
st.title("Story Generator")

genre = st.selectbox(
    "Select Story Genre",
    ["Adventure", "Fantasy", "Mystery", "Comedy", "Sci-Fi"]
)

character = st.selectbox(
    "Select Main Character",
    ["A Student", "A Robot", "A Detective", "A Dragon", "A Princess"]
)

length = st.selectbox(
    "Select Story Length",
    ["Short", "Medium", "Long"]
)

if st.button("Generate Story"):

    prompt = template.invoke(
        {
            "genre": genre,
            "character": character,
            "length": length
        }
    )

    response = model.invoke(prompt)

    st.subheader("Your Story")
    st.write(response.content)