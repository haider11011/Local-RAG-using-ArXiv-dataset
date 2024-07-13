import streamlit as st
from scripts.qa_system import initialize_rag_system, ask_question

st.title("Local RAG System for JSON Data")

json_file = "data/arxiv-metadata-oai-snapshot.json"
rag_chain = initialize_rag_system(json_file)

question = st.text_input("Enter your question:")

if question:
    answer = ask_question(rag_chain, question)
    st.write("Answer:", answer)