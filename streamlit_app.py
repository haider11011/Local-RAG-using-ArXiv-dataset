import streamlit as st
from src.rag import RAG

# Initialize RAG system
rag = RAG(dataset_path="data/arxiv-metadata-oai-snapshot.json")

st.title("Advanced Retrieval-Augmented Generation (RAG) System")

# User input
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Retrieving documents and generating answer..."):
            answer = rag.answer_question(question)
            st.write("### Answer:")
            st.write(answer)
    else:
        st.write("Please enter a question.")