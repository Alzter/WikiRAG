from query_decomposer import QueryDecomposer
from retrieval import Retrieval, DenseRetrieval

class IterativeRetrieval:

    def __init__(self, corpus_path : str):
        self.retriever = Retrieval(corpus_path)
        self.qd = QueryDecomposer()

    def evaluate_similarity(self, sentence_a, sentence_b):
        """
        Evaluate how similar in meaning two sentences are as a scalar from 0 to 1.
        
        Args:
            sentence_a (str): The first sentence.
            sentence_b (str): The second sentence.
        
        Returns:
            similarity (float): The sentence likeness on a scale from 0 (dissimilar) to 1 (similar).
        """
        return DenseRetrieval.compare_documents(
            self.retriever.embedding_model.get_embedding(sentence_a),
            self.retriever.embedding_model.get_embedding(sentence_b)
        )

    def evaluate_answer_confidence(self, response : str):
        """
        Evaluate how confident the model's answer for the question is
        from a scale of 1 (confident) to 0 (not confident).
        """
        # Score how closely the embedding of the model's response aligns with "I don't know".
        uncertainty_score = self.evaluate_similarity(response, "I don't know.")

        # Answer confidence is the inverse of this score
        return 1 - uncertainty_score

    def answer_single_hop_question(self, query : str, max_attempts : int = 5, verbose : bool = True) -> str:
        """
        Answer a single-hop question by retrieving context from Wikipedia.

        Args:
            query (str): The single-hop question to answer.
            max_attempts (int, optional): The number of tries the retrieval network has to successfully find the context for the question.
        
        Returns:
            answer (str): The answer to the question, or "I don't know" if the model could not retrieve the correct context.
        """
        answer = None
        answer_attempts = 0
        visited_articles = []

        while answer is None or answer_attempts < max_attempts:
            print(f"Attempt {answer_attempts + 1} to answer question")

            # Retrieve context for the user's query from a Wikipedia article.
            context, article = self.retriever.get_context(query, num_contexts=1, ignored_articles = visited_articles)

            visited_articles.append(article)

            if verbose: print(f"Retrieved context from article: {article}")
            if verbose: print(f"Attempting to answer question using context: {context}")

            # Attempt to answer the question using the retrieved context.
            answer = self.qd.answer_question_using_context(query, context)
            
            # Evaluate how confident the model is in their answer from 1 to 0.
            answer_confidence = self.evaluate_answer_confidence(answer)
            
            if verbose: print(f"Answer: {answer}\nAnswer confidence:{answer_confidence}")

            # If the model is not confident in their answer, try answering the question again with a new article.
            if answer_confidence < 0.1: answer = None
            answer_attempts += 1
        
        if answer == None: answer = "I don't know."

        return answer

    def answer_multi_hop_question(self, query : str, maximum_reasoning_steps : int = 10):
        """
        Answer a multi-hop question using iterative retrieval.

        Args:
            query (str): The user's query.
            maximum_reasoning_steps (int): 
                How many steps of iterative reasoning (AKA 'hops') the model may make to answer the query.

                E.g., 2 maximum reasoning steps means the model can solve 2-hop questions.
            

        Returns:
            chat_history (list): The chain-of-thought process the model employed to acquire the answer.
            answer (str): The answer to the question, or "I don't know" if the model could not answer.
        """

        # Extract first sub-question, or "That's enough" if context is sufficient
        chat_history, sub_question = self.qd.decompose_question_step(query)

        hops = 0
        
        while self.evaluate_similarity(sub_question, "That's enough.") < 0.9 and hops < maximum_reasoning_steps:

            # Answer the first sub-question using by retrieving context from the knowleddge base.
            sub_answer = self.answer_single_hop_question(sub_question, max_attempts=10)
            
            if sub_answer == "I don't know.": return chat_history, "I don't know."

            # Add the sub-answer to the chat history.
            chat_history.append({'role': 'user', 'content': sub_answer})

            # Extract further sub-questions, or "That's enough" if context is sufficient
            chat_history, sub_question = self.qd.decompose_question_step(chat_history)
            hops += 1

        if hops == maximum_reasoning_steps: return chat_history, "I don't know."

        # TODO: Make model generate final answer using contexts.

        return chat_history, "TODO: Get final answer"

