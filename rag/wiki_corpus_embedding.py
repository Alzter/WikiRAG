import numpy as np
import json, os
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from glob import glob

import sys; sys.path.append("processor")
from document_embedding import DocumentEmbedding

import unicodedata
import re
from tqdm import tqdm
import math

class WikiCorpusEmbedding(DocumentEmbedding):
    """
    Class which can convert a raw-text corpus of Wikipedia into an embedding database.
    """

    def __init__(self, fast : bool = False):
        """
        Args:
            fast (bool): If True, quantizes the embedding model. This leads to faster embedding time, but worse embeddings.
        """
        return super().__init__(fast=fast)

    def read_input_text_files(self, files : list):
        """
        Read text content from all JSON files in a folder and returns it as an array where each element represents a JSON entry in each file.
        Used to extract all Wikipedia articles from a raw text Wikipedia dump. Every entry of the array is a Wikipedia article.
        
        Args:
            files (list): A list of all Wikipedia raw text dump file paths to load.
        
        Returns:
            all_text (list): All text from the folder as an array for each article.
        """
        
        text_list = []

        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Parse each line as JSON and extract the 'text' field
                        data = json.loads(line)
                        text_content = data.get('text', '').strip()  # Strip any leading/trailing whitespace
                        title = data.get('title', '').strip()

                        # Only add articles with text in them.
                        if not text_content: continue
                        if not title: continue

                        # Ignore articles which are disambiguation pages.
                        if 'disambiguation' in title.lower().strip(): continue
                        # text_subset = text_content[:200]
                        # disambig_sentences = ["may refer to", "may also refer to", "can refer to"]

                        # article_is_disamiguation_page = False
                        # for sentence in disambig_sentences:
                        #     if sentence in text_subset:
                        #         article_is_disamiguation_page = True
                        
                        # if article_is_disamiguation_page: continue
                        
                        text_list.append(text_content)

                    except json.JSONDecodeError:
                        continue  # Skip lines that are not valid JSON
        
        return text_list

    def parse_wikipedia_corpus(self, corpus):
        """
        Given a list of Wikipedia articles as raw text, parse the articles into a data structure:

        Args:
            corpus (list[str]): The list of article bodies.
        
        Returns:
            articles (list[dict]): A list of formatted articles using the following structure:
                {'title' : 'Article title', 'summary': 'Sentence summarising the article.', 'paragraphs' : list[str] of all body paragraphs, summary included.}
        """
        articles = []
        
        for article in corpus:

            try:
                paragraphs = self.chunk_text(article)

                # Ignore articles which do not have body text.
                if len(paragraphs) <= 1:
                    print(f"Ignoring article: {article[:30]}...")
                    continue

                # assert((len(paragraphs)>1), "Article must have more than 2 paragraphs.")

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
                print(f"Error parsing Wikipedia article. Traceback: {str(e)}\nArticle text:\n{article[:30]}...")

        return articles
    
    def split_corpus_into_batches(self, raw_text_corpus_path : str, batch_size_mb : 50):
        """
        Given a raw-text corpus created by WikiExtractor,
        split the corpus into a number of batches where each batch
        consumes no more megabytes in size than ``batch_size_mb``.

        Args:
            raw_text_corpus_path (str): The path of the raw text corpus to read.
            batch_size_mb (int): How many megabytes worth of Wikipedia articles to contain within a single batch.

        Returns:
            batches (list): A list of batches of all Wikipedia articles.
            Each batch itself is a list[str] containing the file paths of all Wikipedia articles in the batch.
        """
        
        # Get all Wikipedia files within the batch.
        files = glob(raw_text_corpus_path + "//*//wiki_*")

        if len(files) == 0: raise FileNotFoundError(f"No Wikipedia extracted raw text articles found in directory {raw_text_corpus_path}")

        # Get the size of each Wikipedia dump in MB
        file_size_mb = math.ceil(os.stat(files[0]).st_size / pow(1024, 2))

        # We can't have a batch size smaller than the smallest file size.
        if batch_size_mb < file_size_mb:
            batch_size_mb = file_size_mb
            print("Warning: Batch size was set to {batch_size_mb} MB." + \
                  "This is lower than each file's size of {file_size_mb} MB." + \
                  "We can't have a batch size smaller than the smallest file size, so " + \
                  "The batch size has been set to {file_size_mb} MB.")

        # Calculate the number of files which can be included in every batch
        files_per_batch = batch_size_mb // file_size_mb

        # Calculate the number of batches we need to cover the entire dump
        num_batches = math.ceil(len(files) / files_per_batch)

        batches = [files[i:i+files_per_batch] for i in range(num_batches)]

        return batches

    def embed_wikipedia_raw_text(self, raw_text_corpus_path : str, 
                                output_dir : str,
                                batch_size_mb : 50,
                                fast : bool = False,
                                overwrite : bool = False):
        """
        Converts a raw text Wikipedia knowledge corpus into a RAG knowledge base and stores it in ``output_dir``.

        Args:
            raw_text_corpus_path (str): The path of the raw text corpus to read.
            output_dir (str): The directory where the embeddings will be saved.
            batch_size_mb (int): How many megabytes worth of Wikipedia articles to embed at a time.
            fast (bool, optional): If true, quantizes the embedding model, leading to faster embeddings but worse accuracy.
            overwrite (bool, optional): If true, overwrites existing entries in the knowledge base. Defaults to False.
            
        Returns:
            output_dir (str): The directory where the embeddings were saved.
        """
        
        batches = self.split_corpus_into_batches(raw_text_corpus_path, batch_size_mb=batch_size_mb)

        print("----------------------------------")
        print(f"Corpus has been split into {len(batches)} batches.")
        print("----------------------------------")

        progress_bar = tqdm("Embedding articles...", total=len(batches), unit="article")

        for batch_id, batch in enumerate(batches):

            article_json = self.read_input_text_files(batch)

            articles = self.parse_wikipedia_corpus(article_json)

            if not os.path.exists(output_dir): os.makedirs(output_dir)

            for article in articles:
                
                progress_bar.update(1 / len(articles))
                progress_bar.set_postfix_str(f"{article['title'][:10]}... - Batch {batch_id + 1}/{len(batches)}")
                progress_bar.refresh()

                self.create_document_embedding(
                    article['title'],
                    article['summary'],
                    article['paragraphs'],
                    output_dir=output_dir,
                    overwrite=overwrite)

        progress_bar.close()

        return output_dir
