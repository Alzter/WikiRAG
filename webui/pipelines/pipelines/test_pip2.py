"""
title: RAG
author: Alexander Small, Toan Nguyen, Matthew Crick
date: 2024-10-19
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings from a GitHub repository.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-readers-github
"""

from typing import List, Union, Generator, Iterator
import os
import asyncio
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
# from transformer_model import TransformerModel
from abc import ABC

class TransformerModel(ABC):
    """
    Abstract class which instantiates an embedding and tokenizer model using the ``transformers`` library when it is created.
    Model name and quantization are configurable in the ``__init__`` method parameters.
    """

    def __init__(self, model_name : str, causal : bool, quantized : bool, use_gpu : bool =True):
        """
        Create the model and tokenizer needed.
        """
        # Run the device on GPU only if NVIDIA CUDA drivers are installed.
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'

        if self.device == 'cpu': print("No GPU found: using CPU for model.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quantized,
            bnb_4bit_compute_dtype=torch.bfloat16# if quantized else None
        )

        print(f"Loading transformer model and tokenizer from transformers library: {model_name}\nPlease wait...\n")

        model_method = AutoModelForCausalLM if causal else AutoModel
        
        self.model = model_method.from_pretrained(model_name, trust_remote_code=True, device_map=self.device, quantization_config = quantization_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map=self.device, quantization_config = quantization_config)
            
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

class Pipeline:
    def __init__(self):
        self.name = "TEST PIPELINE 2"
        self.llm = LLM()

    async def on_startup(self):
        # self.llm = LLM()
        pass
        

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        # llm = LLM()
        # self.llm = LLM()
        chat_history, answer = self.llm.generate_response(input=user_message, max_new_tokens=100, truncate=True)
        

        return f"This is user input {user_message}\n Answer: {chat_history}"

