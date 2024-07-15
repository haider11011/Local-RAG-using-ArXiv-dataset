from datasets import Dataset
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class Retriever:
    def __init__(self, dataset_path: str):
        self.dataset = Dataset.from_json(dataset_path)
        self.docs = self.dataset.map(lambda x: {"text": x["abstract"]})
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.index = FAISS.from_documents(self.docs["text"], self.embeddings)
    
    def retrieve(self, query: str, top_k: int = 5):
        return self.index.similarity_search(query, k=top_k)