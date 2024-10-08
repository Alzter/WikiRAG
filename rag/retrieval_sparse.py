import k_best
from rank_bm25 import BM25Okapi

class SparseRetrieval():
    """
    Sparse document retrieval technique. This class retrieves documents
    using lexical matching, i.e., the number of exact word matches.

    AKA, an N-gram retrieval method.
    """

    @staticmethod
    def get_scores(query : str, corpus : list[str]):
        """
        Given an input query string and a corpus (a list of document strings),
        evaluate the lexical similarity between each document in the corpus and the query
        on a scale from 0 to 1. Return the list of scores for each corpus document.

        Source: https://pypi.org/project/rank-bm25/

        Args:
            query (str): "windy London"
            corpus (list[str]): ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]

        Returns:
            scores (ndarray[int]): array([0.        , 0.93729472, 0.        ])
        """

        tokenized_corpus = [doc.split(" ") for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.split(" ")

        return bm25.get_scores(tokenized_query)

    @staticmethod
    def get_k_best_documents(k : int, query : str, corpus : list[str]):
        """
        Given an input query, retrieve k most similar documents from the corpus
        using sparse retrieval (lexical matching) in descending order of similarity.

        Args:
            n (int): The number of documents to retrieve. E.g., 1.
            query (str): "windy London"
            corpus (list[str]): ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]

        Returns:
            top_k (list[str]): The k most matching items in the corpus. E.g., ['It is quite windy in London']
        """

        scores = SparseRetrieval.get_scores(query, corpus)

        top_k = k_best.get_k_best(k, corpus, scores)

        return top_k