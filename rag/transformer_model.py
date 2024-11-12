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

        if os.path.exists(model_name):
            # Load model in local if existed
            print(f"Loading {model_name} from local\nPlease wait...\n")

            self.model = AutoModelForCausalLM.from_pretrained(f"./models/Meta-Llama-3.1-8B-Instruct")

            self.tokenizer = AutoTokenizer.from_pretrained("./models/NoInstruct-small-Embedding-v0")

        else:
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            model_method = AutoModelForCausalLM if causal else AutoModel
            
            self.model = model_method.from_pretrained(model_name, trust_remote_code=True, device_map=self.device, quantization_config = quantization_config)#, token=os.getenv('HF_TOKEN'))
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map=self.device,output_hidden_states=True)#, quantization_config = quantization_config)#, token=os.getenv('HF_TOKEN'))
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            self.model.save_pretrained("models/Meta-Llama-3.1-8B-Instruct")
            self.tokenizer.save_pretrained("models/NoInstruct-small-Embedding-v0")  
            
