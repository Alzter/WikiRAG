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
    
    def generate_response(self, input : str | list, max_new_tokens : int, truncation : bool = True, continue_final_message : bool = False):
        """
        Generate a response to the user's query.

        Args:
            input (str/list): Can either be a direct input of data type ``str``, or a chat history of type ``list``.
            max_tokens (int): The maximum number of tokens the LLM is allowed to generate for the response.
            truncation (bool, optional): Whether to force the generated text to be of length max_new_tokens by cutting it off.
            continue_final_message (bool, optional):
            This indicates that you want the model to continue the last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
            By default this is ``True`` when the final message in the input chat has the assistant role and ``False`` otherwise, but you can manually override that behaviour by setting this flag.
        
        Returns:
            chat_history (list): The entire chat history, including the user's prompt and the LLM's response.
            assistant_response (str): The LLM's response to the input.
        """

        chat_history = self.pipeline(
            input,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            truncation = truncation,
            max_new_tokens=max_new_tokens,
            # continue_final_message=continue_final_message
        )

        try:
            chat_history = chat_history[0]['generated_text']
            assistant_response = chat_history[-1]['content']
        except Exception as e:
            print(f"Error retrieving assistant response for LLM generation.\nChat history: {str(chat_history)}\nTraceback:\n{str(e)}")
            assistant_response = None

        return chat_history, assistant_response