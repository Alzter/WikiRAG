import glob, os
import numpy as np
from embedding_model import EmbeddingModel
import k_best
from rank_bm25 import BM25Okapi
import torch

class Document():
    def __init__(self, title : str, summary : str, embedding : torch.Tensor):
        self.title = title; self.summary = summary; self.embedding = embedding

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
        summary_embeddings = glob.glob(f"{corpus_path}/*/summary.npy")

        documents = []

        assert(len(summary_files) == len(summary_embeddings), "Summary text files should directly map to summary embedding files.")
        
        for summary_file, summary_embedding in zip(summary_files, summary_embeddings):
            
            # Get the parent directory of the summary file
            parent_directory = os.path.split(os.path.dirname(summary_file))[-1]

            with open(summary_file, "r") as f:
                summary_text = f.read()

            embedding_data = np.load(summary_embedding, encoding='bytes', allow_pickle=True)
            embedding_data = torch.Tensor(embedding_data)

            document = Document(title=parent_directory, summary=summary_text, embedding=embedding_data)

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

            embedding_data = np.load(embedding_file, encoding='bytes', allow_pickle=True)
            embedding_data = torch.Tensor(embedding_data)

            with open(raw_text_file, "r") as f:
                raw_text = f.read()
            
            embedding = Embedding(raw_text=raw_text, embedding=embedding_data)

            embeddings.append(embedding)
        
        return embeddings
    
    def get_context(self, query : str, num_contexts = 1, use_sparse_retrieval = False, exhaustive = False) -> list[str]:
        """
        Given a user question, retrieve contexts in the form of paragraphs from Wikipedia articles to answer the question.

        Method:
        1. One Wikipedia article is retrieved which best answers the initial query.
        2. Paragraphs are fetched from the chosen Wikipedia article as contexts to the query.

        NOTE: This function **only works for single-hop** questions where the answer can be found from *one* article.
        For multi-hop questions, you must first decompose the question using the ``QueryDecomposition`` class.

        Args:
            query (str): The user's question. E.g., 'Who was the first person to land on the moon?'

            num_contexts (int, optional): How many paragraphs to retrieve. Defaults to 1.

            use_sparse_retrieval (bool, optional):
                Controls whether to use dense or sparse retrieval to find the best Wikipedia article to answer the user's query.

                If True, uses a BM25 search of article raw text summaries to find the Wikipedia article.

                If False, uses cosine similarity search of article summary embeddings to find the Wikipedia article.

            exhaustive (bool, optional):
                If True, skip the article selection step and find context by searching through **all Wikipedia articles**.

                WARNING: This has high time and computational complexity.
        
        Returns:
            context (str | list[str]): A string or list of ``num_context`` contexts, where each context is a paragraph from a Wikipedia article.
            article (str | None): The name of the article context was retrieved from, or ``None`` if ``exhaustive`` was True.

        """
        
        print("Embedding user query for dense retrieval:")

        # Convert query into an embedding
        query_embedding = self.embedding_model.get_embedding(query, input_is_query=True)

        if exhaustive:

            # Load all embeddings into memory (this takes a long time)
            if not hasattr(self, "all_embeddings"):

                print("Loading all document chunk embeddings into memory...")
                self.all_embeddings = self.get_document_embeddings(None)

            embeddings = self.all_embeddings
            article = None
            
        else:
            if use_sparse_retrieval:
                print("Finding best article to use as context with sparse retrieval:")

                # Use BM25 (sparse retrieval) to acquire one Wikipedia article
                # which has the most n-gram lexical matches to the user query.
                best_document = SparseRetrieval.get_k_best_documents(1, query, self.documents)[0]
            else:

                print("Finding best article to use as context with dense retrieval:")

                best_document = DenseRetrieval.get_k_best_documents(1, query_embedding, self.documents)[0]

            article = best_document.title

            print(f"Using Wikipedia article: {best_document.title} for context")

            # Retrieve all chunk embeddings from said document
            embeddings = self.get_document_embeddings(article)

        print(f"Found {len(embeddings)} chunks for within context.")
        print("Finding best article chunks to use as context with dense retrieval:")

        best_embeddings = DenseRetrieval.get_k_best_documents(num_contexts, query_embedding, embeddings)

        print("Context successfully retrieved.")

        retrieved_contexts = [embedding.raw_text for embedding in best_embeddings]

        # Return context as a string if only one context was retrieved
        if len(retrieved_contexts) == 1: retrieved_contexts = retrieved_contexts[0]

        return retrieved_contexts, article

        
