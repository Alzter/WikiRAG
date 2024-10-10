import transformers
import torch
from language_model import LLM
from model_prompts import Prompt

class QueryDecomposer(LLM):

    def __init__(self):
        super().__init__()
    
    def decompose_question_step(self, input : str | list, max_tokens : int = 50):
        """
        Given a multi-hop question, decomposes the question *once* to generate a sub-question, or returns "That's enough" if the LLM believes the question has been fully decomposed.
        To answer multi-hop questions, The ``Llama-3.1-8B-Instruct`` model is deployed with an elicitive Chain-of-Thought prompt from ``prompt.py``, which is created using a template from [LangGPT](https://github.com/langgptai/LangGPT/).

        Args:
            input (str/list): Can either be a multi-hop question of data type ``str``, or an ongoing LLM chat history of type ``list``.
                If ``str``, the input is treated as the initial multi-hop question, E.g., ``"Who was president of the United States in the year that Citibank was founded?"``.
                If ``list``, the input is treated as a subsequent step in the query decomposition.
            max_tokens (int): The maximum number of tokens the LLM is allowed to generate for the sub-question.
        
        Returns:
            chat_history (list): The entire chat history generated from the LLM, which includes both the CoT prompt, the user query, and the LLM's decomposition.
                                This argument can be fed back into the ``decompose_question_step`` function to generate further sub-questions once context has been retrieved and implemented for the initial sub-question.
            sub_question (str): The sub-question extracted from the full ``chat_history``. Context can be retrieved for this question.
        """

        if isinstance(input, str):
            input = [
                {'role':'system','content':Prompt.cot_prompt}, # Question Decomposition Specialist Prompt
                {'role':'user','content':f"Let's break down this complex question: {input}"}
            ]

        chat_history = self.pipeline(
            input,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            truncation = True,
            max_new_tokens=50
        )

        self.generate_response(input, max_new_tokens=50)

        assistant_response = chat_history[0]['generated_text'][-1]
        sub_question = assistant_response['content']

        return chat_history, sub_question