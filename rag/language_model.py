#from typing import Union
import torch
from transformer_model import TransformerModel
import transformers

class LLM(TransformerModel):
    """
    Class which can generate text using an LLM.
    """

    def __init__(self, model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct", causal = True, quantized = True, use_gpu = True):
        super().__init__(model_name, causal, quantized, use_gpu)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    def generate_response(self, input : str | list, max_new_tokens = None):
        """
        Generate a response to the user's query.

        Args:
            input (str/list): Can either be a direct input of data type ``str``, or a chat history of type ``list``.
            max_tokens (int, optional): The maximum number of tokens the LLM is allowed to generate for the response.
        
        Returns:
            chat_history (list): The entire chat history, including the user's prompt and the LLM's response.
        """

        return self.pipeline(
            input,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            truncation = True,
            max_new_tokens=max_new_tokens
        )