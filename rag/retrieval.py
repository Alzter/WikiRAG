import glob, os
import numpy as np
from embedding_model import EmbeddingModel
import k_best
from rank_bm25 import BM25Okapi
import torch

class Document():
    def __init__(self, title : str, summary : str):
        self.title = title; self.summary = summary

class Embedding():
    def __init__(self, raw_text : str, embedding : torch.Tensor):
        self.raw_text = raw_text; self.embedding = embedding

class SparseRetrieval():
    """
    Sparse document retrieval technique. This class retrieves documents
    using lexical matching, i.e., the number of exact word matches.

    AKA, an N-gram retrieval method.
    """

    @staticmethod
    def get_scores(query : str, corpus : list[Document]) -> list[float]:
        """
        Given an input query string and a corpus (a list of document strings),
        evaluate the lexical similarity between each document in the corpus and the query
        on a scale from 0 to 1. Return the list of scores for each corpus document.

        Source: https://pypi.org/project/rank-bm25/

        Args:
            query (str): "windy London"
            corpus (list[Document]): ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]

        Returns:
            scores (ndarray[int]): array([0.        , 0.93729472, 0.        ])
        """

        # Extract summary text all from documents
        corpus = [document.summary for document in corpus]

        tokenized_corpus = [doc.split(" ") for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.split(" ")

        return bm25.get_scores(tokenized_query)

    @staticmethod
    def get_k_best_documents(k : int, query : str, corpus : list[Document]) -> list[Document]:
        """
        Given an input query, retrieve k most similar documents from the corpus
        using sparse retrieval (lexical matching) in descending order of similarity.

        Args:
            n (int): The number of documents to retrieve. E.g., 1.
            query (str): "windy London"
            corpus (list[Document]): ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]

        Returns:
            top_k (list[str]): The k most matching items in the corpus. E.g., ['It is quite windy in London']
        """

        scores = SparseRetrieval.get_scores(query, corpus)

        top_k = k_best.get_k_best(k, corpus, scores)

        return top_k

class DenseRetrieval():
    """
    Dense retrieval technique using document embeddings.
    This class retrieves documents by calculating their
    semantic similarity using a cosine similarity function.
    """

    @staticmethod
    def compare_documents(a : torch.Tensor, b : torch.Tensor) -> float:
        """
        Returns the similarity between embeddings a and b using cosine similarity.
        """
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    @staticmethod
    def get_scores(query : torch.Tensor, corpus : list[Embedding]) -> list[float]:
        """
        Given an input query embedding and a corpus (a list of document embeddings),
        evaluate the semantic similarity between each document in the corpus and the query
        on a scale from 0 to 1. Return the list of scores for each corpus document.
        """
        corpus_embeddings = [document.embedding for document in corpus]

        scores = []

        for embedding in corpus_embeddings:
            similarity = DenseRetrieval.compare_documents(query, embedding)[0]
            scores.append(similarity)
        
        return scores
    
    @staticmethod
    def get_k_best_documents(k : int, query : torch.Tensor, corpus : list[Embedding]) -> list[Embedding]:
        """
        Given an input query, retrieve k most similar embeddings from the corpus
        using dense retrieval (cosine similarity of embeddings) in descending order of similarity.

        Args:
            n (int): The number of documents to retrieve. E.g., 1.
            query (torch.Tensor):
            corpus (list[Embedding]):

        Returns:
            top_k (list[Embedding]): The k most matching items in the corpus.
        """

        scores = DenseRetrieval.get_scores(query, corpus)
        #print(f"Scores: {scores}")
        #print(f"Corpus: {corpus}")
        top_k = k_best.get_k_best(k, corpus, scores)

        return top_k

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
        """
        If document_name == None, load *all* embeddings.
        """

        embeddings = []

        if document_name == None:
            embedding_files = glob.glob(self.corpus_path + "//*//chunk_*.npy")
            embedding_texts = glob.glob(self.corpus_path + "//*//chunk_*.txt")
        else:
            document_embedding_path = os.path.join(self.corpus_path, document_name.strip())

            embedding_files = glob.glob(document_embedding_path + "//chunk_*.npy")
            embedding_texts = glob.glob(document_embedding_path + "//chunk_*.txt")
        
        assert(len(embedding_files) == len(embedding_texts), "Embedding raw text files should directly map to embedding data files.")

        for embedding_file, raw_text_file in zip(embedding_files, embedding_texts):

            # The embedding data is loaded as a numpy array
            embedding_data = np.load(embedding_file, encoding='bytes', allow_pickle=True)

            # Then converted into a Tensor again. Is this efficient? No! Too bad!
            embedding_data = torch.Tensor(embedding_data)

            with open(raw_text_file, "r") as f:
                raw_text = f.read()
            
            embedding = Embedding(raw_text=raw_text, embedding=embedding_data)

            embeddings.append(embedding)
        
        return embeddings
    
    def get_context(self, query : str, num_contexts = 10, use_sparse_retrieval = True) -> list[str]:

        if not use_sparse_retrieval:

            # Load all embeddings into memory (this takes a long time)
            if not hasattr(self, "all_embeddings"):

                print("Loading all document chunk embeddings into memory...")
                self.all_embeddings = self.get_document_embeddings(None)

            embeddings = self.all_embeddings
            
        else:
            print("Finding best article to use as context with sparse retrieval:")

            # Use BM25 (sparse retrieval) to acquire one Wikipedia article
            # which has the most n-gram lexical matches to the user query.
            best_document = SparseRetrieval.get_k_best_documents(1, query, self.documents)[0]

            print(f"Using Wikipedia article: {best_document.title} for context")

            # Retrieve all chunk embeddings from said document
            embeddings = self.get_document_embeddings(best_document.title)
            
            print(f"Found {len(embeddings)} chunks within article.")

            print("Embedding user query for dense retrieval:")

        # Convert query into an embedding
        query_embedding = self.embedding_model.get_embedding(query, input_is_query=True)

        print("Finding best article chunks to use as context with dense retrieval:")

        best_embeddings = DenseRetrieval.get_k_best_documents(num_contexts, query_embedding, embeddings)

        print("Context successfully retrieved.")

        retrieved_contexts = [embedding.raw_text for embedding in best_embeddings]

        return retrieved_contexts

        
