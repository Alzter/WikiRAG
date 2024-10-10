from query_decomposer import QueryDecomposer
from retrieval import Retrieval, DenseRetrieval

class IterativeRetrieval:

    def __init__(self, corpus_path : str):
        qd = QueryDecomposer()
        retriever = Retrieval(corpus_path)

    def evaluate_answer_confidence(self, response : str):
        """
        Evaluate how confident the model's answer for the question is
        from a scale of 1 (confident) to 0 (not confident).
        """
        # Score how closely the embedding of the model's response aligns with "I don't know".
        uncertainty_score = DenseRetrieval.compare_documents(
            self.retriever.embedding_model.get_embedding(response),
            self.retriever.embedding_model.get_embedding("I don't know.")
        )

        # Answer confidence is the inverse of this score
        return 1 - uncertainty_score

    def answer_single_hop_question(self, query : str, max_attempts = int):
        """
        Answer a single-hop question by retrieving context from Wikipedia.

        Args:
            query (str): The single-hop question to answer.
            max_attempts (int): The number of tries the retrieval network has to successfully find the context for the question.
        
        Returns:
            answer (str): The answer to the question, or "I don't know" if the model could not retrieve the correct context.
        """
        answer = None
        answer_attempts = 0
        visited_articles = []

        while answer is None or answer_attempts < max_attempts:

            # Retrieve context for the user's query from a Wikipedia article.
            context, article = self.retriever.get_context(query, num_contexts=1, ignored_articles = visited_articles)

            visited_articles.append(article)

            # Attempt to answer the question using the retrieved context.
            answer = self.qd.answer_question_using_context(
                query,
                context)
            
            # Evaluate how confident the model is in their answer from 1 to 0.
            answer_confidence = self.evaluate_answer_confidence(answer)

            # If the model is not confident in their answer, try answering the question again with a new article.
            if answer_confidence < 0.3: answer = None
            answer_attempts += 1
        
        if answer == None: answer = "I don't know."

        return answer

    def answer_multi_hop_question(self, query : str):


