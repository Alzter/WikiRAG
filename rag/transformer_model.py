import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from abc import ABC
import os

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

        if self.device == 'cuda':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=quantized,
                bnb_4bit_compute_dtype=torch.bfloat16# if quantized else None
            )
        else:
            quantization_config = None

        print(f"Loading transformer model and tokenizer from transformers library: {model_name}\nPlease wait...\n")

        model_method = AutoModelForCausalLM if causal else AutoModel
        
        self.model = model_method.from_pretrained(model_name, trust_remote_code=True, device_map=self.device, quantization_config = quantization_config, token=os.getenv('HF_TOKEN'))
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map=self.device, quantization_config = quantization_config, token=os.getenv('HF_TOKEN'))
    