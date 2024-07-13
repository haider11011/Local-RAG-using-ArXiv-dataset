from langchain.chains import RetrievalQA
from langchain.retrievers import VectorStoreRetriever
from transformers import pipeline
from scripts.create_vector_db import create_vector_database

def initialize_rag_system(json_file):
    vectorstore = create_vector_database(json_file)
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    retriever = VectorStoreRetriever(vectorstore)
    rag_chain = RetrievalQA(retriever=retriever, llm=qa_model)
    return rag_chain

def ask_question(rag_chain, question):
    result = rag_chain({"question": question})
    return result["answer"]