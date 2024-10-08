import numpy as np
import json, os
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

import sys; sys.path.append("processor")
from embedding_model import EmbeddingModel

class CorpusEmbedding(EmbeddingModel):
    """
    Class which can convert a raw-text corpus of Wikipedia into an embedding database.
    """

    def __init__(self):
        return super().__init__()

    def read_input_texts_from_folder(self, raw_text_corpus_path, return_as_string):
        """
        Read text content from all JSON files in a folder and returns it as an array where each element represents a JSON entry in each file.
        Used to extract all Wikipedia articles from a raw text Wikipedia dump. Every entry of the array is a Wikipedia article.
        
        Args:
            raw_text_corpus_path (str): The path of the raw text corpus to read.
            return_as_string (bool): If true, all text is returned concatenated into a single string using line breaks.
        
        Returns:
            all_text (list | str): All text from the folder either as an array for each article or concatenated.
        """
        
        text_list = []

        for root, _, files in os.walk(raw_text_corpus_path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            # Parse each line as JSON and extract the 'text' field
                            data = json.loads(line)
                            text_content = data.get('text', '').strip()  # Strip any leading/trailing whitespace
                            if text_content:  # Ensure only non-empty content is added
                                text_list.append(text_content)
                        except json.JSONDecodeError:
                            continue  # Skip lines that are not valid JSON
        
        if return_as_string: text_list = "\n".join(text_list)
        return text_list

    def parse_wikipedia_corpus(self, corpus):
        
        articles = []
        
        for article in corpus:
            paragraphs = article.split("\n\n")
            try:
                assert((len(paragraphs)>1), "Article must have more than 2 paragraphs.")

                title = paragraphs[0] # Title
                summary = paragraphs[1] # First paragraph of article = summary
                body_paragraphs = paragraphs[1:] # All paragraphs excluding title

                # Create data structure to represent Wikipedia article
                articles.append({
                    "title" : title.strip(),
                    "summary" : summary,
                    "paragraphs" : body_paragraphs
                })

            except Exception as e:
                print(f"Error parsing Wikipedia article. Traceback: {str(e)}\nArticle text:\n{article}")

        return articles
    
    def corpus_to_embeddings(self, raw_text_corpus_path : str, output_dir : str):
        """
        Converts a raw text knowledge corpus into a NumPy array of chunked embeddings and saves the resulting array to ``output_dir``.

        Args:
            raw_text_corpus_path (str): The path of the raw text corpus to read.
            output_dir (str): The directory where the embeddings will be saved.
            
        Returns:
            output_dir (str): The directory where the embeddings were saved.
        """

        wikipedia = self.read_input_texts_from_folder(raw_text_corpus_path, False)
        articles = self.parse_wikipedia_corpus(wikipedia)

        if not os.path.exists(output_dir): os.makedirs(output_dir)

        for article in articles:
            title = article['title']
            summary = article['summary']
            paragraphs = article['paragraphs']

            article_path = os.path.join(output_dir, title.strip())
            print(article_path)

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

                print(f"Embed paragraph {paragraph_id} for article {title}")

        return output_dir
