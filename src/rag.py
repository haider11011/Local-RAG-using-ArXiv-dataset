from src.retriever import Retriever
from src.reader import Reader

class RAG:
    def __init__(self, dataset_path: str, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        self.retriever = Retriever(dataset_path)
        self.reader = Reader(model_name)
    
    def answer_question(self, question: str, top_k: int = 5):
        retrieved_docs = self.retriever.retrieve(question, top_k)
        context = " ".join([doc.page_content for doc in retrieved_docs])
        return self.reader.generate_answer(context, question)