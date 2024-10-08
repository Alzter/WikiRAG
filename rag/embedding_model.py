#from typing import Union
import torch
from transformer_model import TransformerModel

class EmbeddingModel(TransformerModel):
    """
    Class which can embed text using an embedding model.
    """

    def __init__(self, model_name = "avsolatorio/NoInstruct-small-Embedding-v0", causal = False, quantized = True, use_gpu = True):
        """
        Create embedding model. Source: [NoInstruct-small-Embedding-v0](https://huggingface.co/avsolatorio/NoInstruct-small-Embedding-v0).
        """
        return super().__init__(model_name, causal, quantized, use_gpu)
    
    def get_embedding(self, text: str | list[str], input_is_query = False):
        """
        Generate an embedding for a given text. Source: [NoInstruct-small-Embedding-v0](https://huggingface.co/avsolatorio/NoInstruct-small-Embedding-v0).
        
        Args:
            text (str/list[str]): The text or list of texts to be embedded.
            input_is_query (bool): If true, treats the input as a query, otherwise treats the input as a sentence.
                                    The model is optimized to use the mean pooling for queries, while the sentence / document embedding uses the [CLS] representation.
        
        Returns:
            vectors (torch.Tensor): The text embedding. The embedding is of shape ``[1, 384]``.
        """

        self.model.eval() # Set model to evaluation mode.

        if isinstance(text, str): text = [text] # Ensure text is a list.

        inp = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            output = self.model(**inp)

        # The self.model is optimized to use the mean pooling for queries,
        # while the sentence / document embedding uses the [CLS] representation.

        if input_is_query:
            vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
            vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
        else:
            vectors = output.last_hidden_state[:, 0, :]

        return vectors