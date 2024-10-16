from query_decomposer import QueryDecomposer
from retrieval import Retrieval, DenseRetrieval
from model_prompts import Prompt
import re

class IterativeRetrieval:

    def __init__(self, corpus_path : str, num_threads=4):
        print("Loading RAG model...")
        self.retriever = Retrieval(corpus_path, num_threads=num_threads)
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

    def evaluate_answer_confidence(self, response : str) -> float:
        """
        Evaluate how confident the model's answer for the question is
        from a scale of 1 (confident) to 0 (not confident).

        Args:
            response (str): The LLM's response to any given question.

        Returns:
            answer_confidence (float): The model's confidence that they know the answer from 1 to 0.
                                        Any value under 0.1 should be treated as the model not knowing.
        """

        # Sanitise the response by removing non-alphanumeric characters excluding spaces and converting it to lowercase.
        response_sanitised = re.sub(r'[^a-zA-Z0-9|\s]', '', response).lower().strip()

        # Score an immediate fail if the model said "I don't know" at any point
        # NOTE: This may be a bad idea, because there may be answers which need to say "I don't know" for legitimate reasons, such as for quoting somebody.
        if "i dont know" in response_sanitised: return 0

        # Score how closely the embedding of the model's response aligns with "I don't know".
        uncertainty_score = self.evaluate_similarity(response, "I don't know.")

        # Answer confidence is the inverse of this score
        return 1 - uncertainty_score

    def answer_single_hop_question(self, query : str, num_chunks : int = 1, max_attempts : int = 1, use_sparse_retrieval : bool = False, exhaustive_retrieval : bool = False, use_chain_of_thought : bool = False, verbose : bool = True) -> str:
        """
        Answer a single-hop question by retrieving context from Wikipedia.

        Args:
            query (str): The single-hop question to answer.
            num_chunks (int, optional):
                How many paragraphs to retrieve from each Wikipedia article to use as context.
                
                More context chunks result in higher answer accuracy, but greater answer latency.

                Defaults to 1.
            max_attempts (int, optional): The number of tries the retrieval network has to successfully find the context for the question.
            use_sparse_retrieval (bool, optional):
                Controls whether to use dense or sparse retrieval to find the best Wikipedia article to answer the user's query.

                If True, uses a BM25 search of article raw text summaries to find the Wikipedia article.

                If False, uses cosine similarity search of article summary embeddings to find the Wikipedia article.
            exhaustive_retrieval (bool):
                If True, retrieves context using all of the Wikipedia corpus as a search space. If False, only retrieves context from a single Wikipedia article.

                WARNING: Enabling ``exhaustive_retrieval`` increases retrieval latency.
            use_chain_of_thought (bool): Whether to get the LLM to explain their reasoning process as they generate the answer.
                Only set this to True when getting the LLM to generate a final answer for the user's query.
            verbose (bool, optional): If True, prints the answering process to the console.
        Returns:
            answer (str): The answer to the question, or "I don't know" if the model could not retrieve the correct context.
            chat_history (list[dict]): The LLM's reasoning process to answer the question.
            article (str): The name of the article used to answer the question.
            
        """

        answer = None
        answer_attempts = 0
        visited_articles = []

        chat_history = []

        while answer is None and answer_attempts < max_attempts:
            if verbose: print(f"Attempt {answer_attempts + 1} to answer question")

            # Retrieve context for the user's query from a Wikipedia article.
            context, article = self.retriever.get_context(query, num_contexts=num_chunks, ignored_articles = visited_articles, use_sparse_retrieval = use_sparse_retrieval, exhaustive=exhaustive_retrieval)

            # If we retrieve more than one paragraph for context,
            # concatenate the paragraphs together with line breaks.
            if num_chunks > 1: context = "\n\n".join(context)

            visited_articles.append(article)

            if verbose: print(f"Retrieved context from article: {article}")
            if verbose: print(f"Attempting to answer question using context: {context}")

            # Attempt to answer the question using the retrieved context.
            answer, sub_q_chat_history = self.qd.answer_question_using_context(query, context, use_chain_of_thought=use_chain_of_thought)

            chat_history.append(sub_q_chat_history)
            
            # Evaluate how confident the model is in their answer from 1 to 0.
            answer_confidence = self.evaluate_answer_confidence(answer)
            
            if verbose: print(f"Answer: {answer}\nAnswer confidence:{answer_confidence}")

            # If the model is not confident in their answer, try answering the question again with a new article.
            if answer_confidence < 0.1: answer = None
            answer_attempts += 1
        
        if answer == None: answer = "I don't know."

        return answer, chat_history, article

    def is_answer_attainable(self, query : str, contexts : list[str]) -> tuple[bool, list[dict]]:
        """
        Given a multi-hop query and a list of contexts, assess whether the query is answerable given the contexts.

        Args:
            query (str): The user's multi-hop query.
            contexts (list[str]): A list of retrieved answers for the question.
        
        Returns:
            question_is_answerable (bool): Whether the question can be answered given the contexts.
            chat_history (list): The model's response / reasoning process.
        """

        context_inputs = [{'role':'user','content':f'Context: {context}'} for context in contexts]

        chat_history = [
            {'role':'system','content':Prompt.is_decomposition_needed},
            {'role':'user','content':f'Question: {query}'},
            *context_inputs,
            {'role':'assistant','content':"Are follow-up questions needed here: "}
        ]

        chat_history, answer = self.qd.generate_response(chat_history, max_new_tokens = 10)

        # Extract only the first word from the answer
        answer = answer.split(" ")[0]

        confidence = self.evaluate_similarity(answer, "Yes")

        question_is_answerable = False if confidence > 0.95 else True

        return question_is_answerable, chat_history
    
    def answer_multi_hop_question_using_context(self, query : str, contexts : list[str], verbose : bool = False):
        """
        Answer a given multi-hop question after context has already been provided for it.
        
        Args:
            query (str): The original multi-hop question.
            contexts (list[str]): A list of retrieved contexts for the question.
        
        Returns:
            final_answer (str): A concise answer to the original question,
                                or "I don't know" if the contexts do not answer the original question.
            chat_history (str): The LLM's internal process of answering the question.
        """

        if verbose: print("Getting LLM to acquire answer using CoT...")
        # Use Chain-Of-Thought to assist the LLM with getting the answer by getting them to think out loud.
        # Getting the LLM to explain their answer at every stage of reasoning yields more accurate answers.
        answer_with_reasoning, reasoning_history = self.qd.answer_question_using_context(query, contexts, use_chain_of_thought=True, max_tokens=300)

        answer_confidence = self.evaluate_answer_confidence(answer_with_reasoning)

        if verbose: print(f"Generated verbose answer:\n{answer_with_reasoning}")

        if answer_confidence < 0.1:
            return "I don't know.", reasoning_history

        # Now that we have the answer, we need to simplify it down into a single sentence,
        # because the user does not need the LLM's reasoning chain.

        if verbose: print("Simplifying answer...")
        # LLM prompt to simplify answer to a single sentence.
        simplification_history = [
            {"role":"system","content":Prompt.simplify_answer},
            {"role":"user","content":f"Input: {answer_with_reasoning}"},
            {"role":"user","content":f"Question: {query}"}
        ]

        if verbose: print(f"Generated concise answer: {answer_concise}")
        simplification_history, answer_concise = self.qd.generate_response(simplification_history, max_new_tokens=100)

        chat_history = reasoning_history.copy().append(simplification_history)

        return answer_concise, chat_history

    def answer_multi_hop_question(self, query : str, maximum_reasoning_steps : int = 5, max_sub_question_answer_attempts : int = 1, num_chunks : int = 1, verbose : bool = True):
        """
        Answer a multi-hop question using iterative retrieval.

        Args:
            query (str): The user's query.
            maximum_reasoning_steps (int): 
                How many steps of iterative reasoning (AKA 'hops') the model may make to answer the query.

                E.g., 2 maximum reasoning steps means the model can solve 2-hop questions.
            max_sub_question_answer_attempts (int):
                How many attempts the model may make at answering each sub-question using retrieved contexts.
                If the model fails to answer any sub-question, the model will end the retrieval process and answer "I don't know".
            num_chunks (int, optional):
                How many paragraphs to retrieve from each Wikipedia article to use as context.
                
                More context chunks result in higher answer accuracy, but greater answer latency.

                Defaults to 1.
            verbose (bool, optional): If true, prints the answering process to the console.

        Returns:
            answer (str): The answer to the question, or "I don't know" if the model could not answer.
            chat_history (list[dict]): The LLM's reasoning process history which was used to acquire the final answer.
            articles (list[str]): The names of all articles used as contexts for the answer.
        """
        
        # If the assistant cannot answer the question satisfactorily as a single-hop question, try decomposing the question.
        # Extract first sub-question.
        if verbose: print(f"Decomposing question: {query}")

        # Begin query decomposition using a variable QD history to keep track of its progress
        qd_history, sub_question = self.qd.decompose_question_step(query)
        
        # Create a separate variable 'full chat history' to keep track of every step of the reasoning process (not just QD)
        full_chat_history = qd_history

        hops = 0

        contexts = []
        articles = []
        
        while self.evaluate_similarity(sub_question, "That's enough.") < 0.9 and hops < maximum_reasoning_steps:        
            if verbose: print(f"Extracted sub-question: {sub_question}")
            if verbose: print("Attempting to answer sub-question...")

            # Answer the first sub-question using by retrieving context from the knowleddge base.
            sub_answer, sub_answer_history, sub_answer_article = self.answer_single_hop_question(sub_question, max_attempts=max_sub_question_answer_attempts, num_chunks=num_chunks)

            # Keep track of what contexts we are referencing for our answer.
            articles.append(sub_answer_article)

            # Add the reasoning process for the LLM's sub-answer to the full chat history.
            full_chat_history.append(sub_answer_history)
            
            # If we could not find appropriate context to answer a sub-question,
            # end the retrieval process and answer "I don't know".
            if sub_answer == "I don't know.":
                full_chat_history.append({"role":"assistant", "content":"I don't know."})
                return "I don't know.", full_chat_history, articles
            
            contexts.append(sub_answer)

            if verbose: print(f"Extracted answer to sub-question: {sub_answer}")

            # Add the sub-answer to the QD history *and* the full chat history.
            qd_history.append({'role': 'user', 'content': sub_answer})
            full_chat_history.append({'role': 'user', 'content': sub_answer})

            # Stop if context is sufficient to answer original question.

            if verbose: print("Evaluating whether contexts are sufficient to answer original query...")
            can_answer, answer_evaluation = self.is_answer_attainable(query, contexts)

            # Add the LLM's assessment of whether the contexts are sufficient to answer the question to the full chat history.
            full_chat_history.append(answer_evaluation)

            if can_answer == True:
                if verbose: print("Model is confident that it can answer the original question")
                break

            # Extract further sub-questions, or "That's enough" if context is sufficient
            if verbose: print(f"Decomposing question again...")

            qd_history, sub_question = self.qd.decompose_question_step(qd_history)

            # Add only the latest sub-question to the full history.
            full_chat_history.append(qd_history[-1])
            hops += 1

        # If we were not able to find enough context to answer the multi-hop question, answer "I don't know".
        if hops == maximum_reasoning_steps:
            full_chat_history.append({"role":"assistant", "content":"I don't know."})
            return "I don't know.", full_chat_history, articles

        if verbose: print(f"Retrieved enough context to answer original question: {query}\nContext:\n{str(contexts)}")

        final_answer, final_answer_retrieval_process = self.answer_multi_hop_question_using_context(query, contexts, verbose = verbose)

        full_chat_history.append(final_answer_retrieval_process)

        return final_answer, full_chat_history, articles


    def answer_question(self, query : str, maximum_reasoning_steps : int = 5, max_sub_question_answer_attempts : int = 1, num_chunks : int = 1, verbose : bool = True):
        """
        Answer a multi-hop question by first trying to answer as a single-hop, then using iterative retrieval if answer was not found.

        Args:
            query (str): The user's query.
            maximum_reasoning_steps (int): 
                How many steps of iterative reasoning (AKA 'hops') the model may make to answer the query.

                E.g., 2 maximum reasoning steps means the model can solve 2-hop questions.
            max_sub_question_answer_attempts (int):
                How many attempts the model may make at answering each sub-question using retrieved contexts.
                If the model fails to answer any sub-question, the model will end the retrieval process and answer "I don't know".
            num_chunks (int, optional):
                How many paragraphs to retrieve from each Wikipedia article to use as context.
                
                More context chunks result in higher answer accuracy, but greater answer latency.

                Defaults to 1.
            verbose (bool, optional): If true, prints the answering process to the console.

        Returns:
            answer (str): The answer to the question, or "I don't know" if the model could not answer.
            chat_history (list[dict]): The LLM's reasoning process history which was used to acquire the final answer.
            articles (list[str]): The names of all articles used as contexts for the answer.
        """

        # Attempt to first answer the question as a single-hop question.
        if verbose: print("Attempting to answer the question directly:")
        attempt_answer, attempt_answer_reasoning, attempt_answer_article = self.answer_single_hop_question(query, num_chunks=3, max_attempts=1, use_sparse_retrieval=False, use_chain_of_thought=False)
        attempt_answer_confidence = self.evaluate_answer_confidence(attempt_answer)

        if verbose: print(f"Attempted answer: {attempt_answer}")
        if verbose: print(f"attempt_answer_confidence: {attempt_answer_confidence}")

        if attempt_answer_confidence > 0.1:
            if verbose: print(f"Model believes attempted answer is correct.")
            return attempt_answer, attempt_answer_reasoning, attempt_answer_article
        
        # If this fails, answer the question using iterative reasoning.
        if verbose: print(f"Attempted answer is not sufficient to answer question.")
        return self.answer_multi_hop_question(query, maximum_reasoning_steps, max_sub_question_answer_attempts, num_chunks, verbose)