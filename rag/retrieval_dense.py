import torch
import k_best
from retrieval import Embedding

class DenseRetrieval():
    """
    Dense retrieval technique using document embeddings.
    This class retrieves documents by calculating their
    semantic similarity using a cosine similarity function.
    """

    @staticmethod
    def compare_documents(a : torch.Tensor, b : torch.Tensor):
        """
        Returns the similarity between embeddings a and b using cosine similarity.
        """
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    @staticmethod
    def get_scores(query : torch.Tensor, corpus : list[Embedding]):
        """
        Given an input query embedding and a corpus (a list of document embeddings),
        evaluate the semantic similarity between each document in the corpus and the query
        on a scale from 0 to 1. Return the list of scores for each corpus document.
        """
        corpus_embeddings = [document.embeddding for document in corpus]

        scores = []

        for embedding in corpus_embeddings:
            similarity = DenseRetrieval.compare_documents(query, embedding)
            scores.append(similarity)
        
        return scores
    
    @staticmethod
    def get_k_best_documents(k : int, query : torch.Tensor, corpus : list[Embedding]):
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

        top_k = k_best.get_k_best(k, corpus, scores)

        return top_k