from rag.language_model import LLM

llm = LLM(quantized=False)

llm.generate_response("Hello?", max_new_tokens=50)