from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-mini")

st.title("📚 Research Paper Summarizer")

# Select Research Paper
paper_input = st.selectbox(
    "Select Research Paper Name",
    (
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "GPT-4 Technical Report",
        "Diffusion Models Beat GANs on Image Synthesis",
        "Retrieval-Augmented Generation (RAG)",
        "LLaMA: Open and Efficient Foundation Language Models",
        "Chain of Thought Prompting",
        "Vision Transformer (ViT)",
        "Deep Residual Learning for Image Recognition (ResNet)"
    )
)

# Select Explanation Style
style_input = st.selectbox(
    "Select Explanation Style",
    (
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical",
        "Interview Preparation",
        "Researcher's View"
    )
)

# Select Explanation Length
length_input = st.selectbox(
    "Select Explanation Length",
    (
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (Detailed explanation)"
    )
)

if st.button("Summarize"):

    prompt = f"""
    Explain the research paper "{paper_input}".

    Explanation Style: {style_input}

    Explanation Length: {length_input}

    Include:
    1. Problem Statement
    2. Main Idea
    3. Architecture/Methodology
    4. Key Contributions
    5. Advantages
    6. Limitations
    7. Real-world Applications
    8. Conclusion

    Use headings and bullet points wherever appropriate.
    """

    with st.spinner("Generating Summary..."):
        result = model.invoke(prompt)

    st.subheader("📖 Summary")
    st.write(result.content)