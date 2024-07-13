from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from scripts.extract_text import extract_text_from_json

def create_vector_database(json_file):
    abstracts = extract_text_from_json(json_file)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS(embedding_model)
    vectorstore.add_texts(abstracts)
    return vectorstore