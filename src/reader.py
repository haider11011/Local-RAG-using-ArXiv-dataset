from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class Reader:
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=bnb_config)
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
    
    def generate_answer(self, context: str, question: str):
        return self.qa_pipeline(question=question, context=context)["answer"]