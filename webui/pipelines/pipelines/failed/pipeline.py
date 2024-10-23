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
import sys;sys.path.append("../ProjectCSurvival/rag")
from wikipedia_corpus_download import WikipediaDownload
from wiki_corpus_embedding import WikiCorpusEmbedding
from pdf_embedding import PDFEmbedding
from iterative_retrieval import IterativeRetrieval
from language_model import LLM

# from retrieval import Retrieval

class Pipeline:
    def __init__(self):
        self.name = "COS30018 RAG"

    async def on_startup(self):

        pass
       

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        llm = LLM()

        chat_history, answer = llm.generate_response(input=user_message, max_new_tokens=100, truncate=True)
        
        return {
            "answer" : answer,
            "reasoning" : chat_history
        }
        return response.response_gen
