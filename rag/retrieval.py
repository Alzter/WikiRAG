import glob, os
import numpy as np
from retrieval_dense import DenseRetrieval
from retrieval_sparse import SparseRetrieval
from embedding_model import EmbeddingModel
import k_best
from rank_bm25 import BM25Okapi
from retrieval import Document
import torch

class Document():
    def __init__(self, title : str, summary : str):
        self.title = title; self.summary = summary

class Embedding():
    def __init__(self, raw_text : str, embeddding):
        self.raw_text = raw_text; self.embeddding = embeddding


class Retrieval():
    
    def __init__(self, corpus_path : str):
        self.documents = self.get_document_summaries(corpus_path)
        self.corpus_path = corpus_path
        self.embedding_model = EmbeddingModel()


    def get_document_summaries(self, corpus_path : str) -> list[Document]:

        """
        Given a corpus of documents, extract the document summaries for all documents into a data structure.
        """

        # Get all summary files in the corpus.
        summary_files = glob.glob(f"{corpus_path}/*/summary.txt")

        documents = []
        
        for summary_file in summary_files:
            
            # Get the parent directory of the summary file
            parent_directory = os.path.split(os.path.dirname(summary_file))[-1]

            with open(summary_file, "r") as f:
                summary_text = f.read()

            document = Document(title=parent_directory, summary=summary_text)

            documents.append(document)
    
        return documents

    def get_document_embeddings(self, document_name) -> list[Embedding]:
        document_embedding_path = os.path.join(self.corpus_path, document_name.strip())

        embeddings = []

        embedding_files = glob.glob(document_embedding_path + "//chunk_*.npy")
        embedding_texts = glob.glob(document_embedding_path + "//chunk_*.txt")
        
        assert(len(embedding_files) == len(embedding_texts), "Embedding raw text files should directly map to embedding data files.")

        for embedding_file, raw_text_file in zip(embedding_files, embedding_texts):
            embedding_data = np.load(embedding_file, encoding='bytes', allow_pickle=True)

            with open(raw_text_file, "r") as f:
                raw_text = f.read()
            
            embedding = Embedding(raw_text=raw_text, embedding=embedding_data)

            embeddings.append(embedding)
        
        return embeddings
    
    def retrieve_context(self, query : str, num_contexts = 10) -> list[str]:
        
        # Use BM25 (sparse retrieval) to acquire one Wikipedia article
        # which has the most n-gram lexical matches to the user query.
        best_document = SparseRetrieval.get_k_best_documents(1, query, self.documents)

        # Retrieve all chunk embeddings from said document
        embeddings = self.get_document_embeddings(best_document.title)

        # Convert query into an embedding
        query_embedding = self.embedding_model.get_embedding(query, input_is_query=True)

        best_embeddings = DenseRetrieval.get_k_best_documents(num_contexts, query_embedding, embeddings)

        retrieved_contexts = [embedding.raw_text for embedding in best_embeddings]

        return retrieved_contexts

        
