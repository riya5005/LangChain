from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-5")

st.header('Research Tool')
user_input=st.text_input("Enter your Prompt")
if st.button('Summarize'):
    result=model.invoke(user_input)
    st.text(result.content)


#output :

# ---------------------------------------
#         Research Tool
# ---------------------------------------

# Enter your Prompt
# +-----------------------------------+
# |                                   |
# +-----------------------------------+

#         [ Summarize ]

# and clicks Summarize, the page updates to:

# ---------------------------------------
#         Research Tool
# ---------------------------------------

# Enter your Prompt
# +-----------------------------------+
# | What is Python?                   |
# +-----------------------------------+

#         [ Summarize ]

# Python is a high-level, general-purpose programming
# language known for its readability and versatility.
# It is widely used in web development, data science,
# artificial intelligence, automation, and more.