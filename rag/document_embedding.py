import numpy as np
import json, os
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from glob import glob
from embedding_model import EmbeddingModel
import re

import sys; sys.path.append("rag")

class DocumentEmbedding(EmbeddingModel):
    def __init__(self, fast : bool = False):
        """
        Args:
            fast (bool): If True, quantizes the embedding model. This leads to faster embedding time, but worse embeddings.
        """
        return super().__init__(quantized=fast)
    
    def chunk_text(self, text : str, delimiter : str = "\n\n"):
        """
        Splits text into chunks, where every chunk is a separate paragraph.
        """

        return text.split(delimiter)

    def sanitise_string(self, input_string : str):
            """
            Sanitises a string to make it usable as a folder name
            by removing all non alphanumeric and whitespace characters.
            """
            return re.sub(r'[^a-zA-Z0-9|\s]', '', input_string).strip()
    
    def create_document_embedding(self, title : str, summary : str, paragraphs : list[str], output_dir : str, overwrite : bool = True):
            """
            Given a document ``title``, ``summary``, and body text ``paragraphs``, save the summary and each
            paragraph inside a folder in ``output_dir`` with name ``title``.

            All text, both summary and paragraphs, are saved as both UTF-8 encoded ``.txt`` files and as embeddings in ``.npy`` files.

            Args:
                title (str): The title of the document.
                summary (str): A consise description of the document. For reference, see (Wikipedia:Short_Description)[https://en.wikipedia.org/wiki/Wikipedia:Short_description]
                paragraphs (list[str]): A list of paragraphs for the document. Each paragraph is embedded separately.
                output_dir (str): The directory to create the folder where the embeddings will be saved.
                overwrite (bool): Whether to overwrite the document embeddings if they already exist.
            
            Returns:
                article_path (str): The folder where the embeddings were saved.
            """

            if not os.path.exists(output_dir): os.makedirs(output_dir)
            
            folder_name = self.sanitise_string(title)
            article_path = os.path.join(output_dir, folder_name)
            # print(f"Embedding article: {article_path}")
            
            if not overwrite:
                
                # Ignore articles we have already saved.
                if os.path.exists(article_path):

                    # Does the summary.txt file exist? If not, don't ignore the article.
                    if os.path.isfile( os.path.join(article_path, "summary.txt") ):
                        return

            if not os.path.exists(article_path): os.mkdir(article_path)

            summary_file_path = os.path.join(article_path, "summary.txt")

            # Write the summary file.
            summary_file = open(summary_file_path, "wb")
            summary_file.write(summary.encode("utf-8"))
            summary_file.close()
            
            # Embed each summary using the embedding model
            summary_embedding = self.get_embedding(summary, input_is_query=False)
            summary_embedding_file_path = os.path.join(article_path, "summary.npy")

            np.save(summary_embedding_file_path, summary_embedding, allow_pickle=True)

            # Treat each paragraph as a chunk
            for paragraph_id, paragraph in enumerate(paragraphs):
                
                # Embed each paragraph using the embedding model
                embedding = self.get_embedding(paragraph, input_is_query=False)

                # Save the raw text of the paragraph into a text file
                paragraph_raw_text_file_path = os.path.join(article_path, f"chunk_{paragraph_id}.txt")
                paragraph_raw_text_file = open(paragraph_raw_text_file_path, "wb")
                paragraph_raw_text_file.write(paragraph.encode("utf-8"))
                paragraph_raw_text_file.close()

                # Save the embedding of the paragraph into a numpy file
                embedding_data_file_path = os.path.join(article_path, f"chunk_{paragraph_id}.npy")
                np.save(embedding_data_file_path, embedding, allow_pickle=True)

                # print(f"Embed paragraph {paragraph_id} for article {title}")

            return article_path