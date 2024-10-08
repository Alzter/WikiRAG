import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from abc import ABC

class LanguageModel(ABC):
    """
    Abstract class which instantiates an embedding and tokenizer model using the ``transformers`` library when it is created.
    Model name and quantization are configurable in the ``__init__`` method parameters.
    """

    def __init__(self, model_name, causal, quantized, use_gpu=True):
        """
        Create the model and tokenizer needed.
        """
        # Run the device on GPU only if NVIDIA CUDA drivers are installed.
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quantized,
            bnb_4bit_compute_dtype=torch.bfloat16 if quantized else None
        )

        model_method = AutoModelForCausalLM if causal else AutoModel
        
        self.model = model_method.from_pretrained(model_name, trust_remote_code=True, device_map='cuda', quantization_config = quantization_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map='cuda', quantization_config = quantization_config)
    