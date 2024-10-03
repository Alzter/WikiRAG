#from typing import Union
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from language_model import LanguageModel

class EmbeddingModel(LanguageModel):

    def __init__(self, model_name = "avsolatorio/NoInstruct-small-Embedding-v0", quantized = False):
        """
        Create embedding model: [NoInstruct-small-Embedding-v0](https://huggingface.co/avsolatorio/NoInstruct-small-Embedding-v0)
        """
        return super().__init__(model_name, quantized)
    
    # Source: https://huggingface.co/avsolatorio/NoInstruct-small-Embedding-v0
    def get_embedding(self, text: str | list[str], input_is_query = False):
        """
        Generate an embedding for a given text.
        
        Args:
            text (str/list[str]): The text or list of texts to be embedded.
            input_is_query (bool): If true, treats the input as a query, otherwise treats the input as a sentence.
                                    The model is optimized to use the mean pooling for queries, while the sentence / document embedding uses the [CLS] representation.
        
        Returns:
            vectors (list): The text embedding.
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