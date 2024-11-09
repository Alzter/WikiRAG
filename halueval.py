# ==================================================================================================================
# HaluEval: A Hallucination Evaluation Benchmark for LLMs
#
# This code was originally written by:
# Junyi Li and Xiaoxue Cheng and Wayne Xin Zhao and Jian-Yun Nie and Ji-Rong Wen in 2023
# Journal={arXiv preprint arXiv:2305.11747}, URL={https://arxiv.org/abs/2305.11747}
#
# Retrieved from:
# https://github.com/RUCAIBox/HaluEval on 11/10/2024
#
# Licensed under the Apache License 2.0 (see LICENSE.TXT for details)

import random
import os
import json
import argparse
import tiktoken
from rag.iterative_retrieval import IterativeRetrieval
from rag.language_model import LLM


class HaluEval():
    def __init__(self, kb_path : str, max_tokens : int = 100, quantized = True):
        from rag.iterative_retrieval import IterativeRetrieval
        self.rag = IterativeRetrieval(kb_path, quantized=quantized)

        self.llm = LLM(quantized=quantized)

        self.max_tokens = max_tokens

    def get_rag_response(self, query : str):
        try:
            answer, _, _ = self.rag.answer_multi_hop_question(query=query, max_new_tokens=self.max_tokens)
            return answer
        except Exception as e:
            print(f"Error generating response to query: {query}\nTraceback: {str(e)}")
            return "I don't know."

    def get_llm_response(self, query : str | list[dict]):
        _, assistant_response = self.llm.generate_response(query, max_new_tokens=self.max_tokens)
        return assistant_response

    def get_qa_response(self, model, question, answer, instruction):
        message = [
            {"role": "system", "content":"You are a hallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""},
            {"role": "user", "content": instruction +
                                        "\n\n#Question#: " + question +
                                        "\n#Answer#: " + answer +
                                        "\n#Your Judgement#: "} 
        ]
        prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
        
        if model == 'rag':
            response = self.get_rag_response(prompt)
        else: self.get_llm_response(message)
        
        return response


    def get_dialogue_response(self, model, dialog, response, instruction):
        message = [
            {"role": "system", "content": "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
            {"role": "user", "content": instruction +
                                        "\n\n#Dialogue History#: " + dialog +
                                        "\n#Response#: " + response +
                                        "\n#Your Judgement#: "}
        ]
        prompt = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
        
        if model == 'rag':
            response = self.get_rag_response(prompt)
        else: self.get_llm_response(message)

        return response


    def num_tokens_from_message(self, message, model="davinci"):
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(message))
        return num_tokens


    def truncate_message(self, prompt1, prompt2, model="davinci"):
        if self.num_tokens_from_message(prompt1 + prompt2, model) > 2033:
            truncation_length = 2033 - self.num_tokens_from_message(prompt2)
            while self.num_tokens_from_message(prompt1) > truncation_length:
                prompt1 = " ".join(prompt1.split()[:-1])
        prompt = prompt1 + prompt2
        return prompt


    def get_summarization_response(self, model, document, summary, instruction):
        message = [
            {"role": "system", "content": "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
            {"role": "user", "content": instruction +
                                        "\n\n#Document#: " + document +
                                        "\n#Summary#: " + summary +
                                        "\n#Your Judgement#: "}
        ]
        prompt1 = instruction + "\n\n#Document#: " + document
        prompt2 = "\n#Summary#: " + summary + "\n#Your Judgement#:"
        prompt = prompt1 + prompt2
        
        if model == 'rag':
            response = self.get_rag_response(prompt)
        else: self.get_llm_response(message)

        return response


    def evaluation_qa_dataset(self, model, file, instruction, output_path):
        with open(file, 'r', encoding="utf-8") as f:
            data = []
            for line in f:
                data.append(json.loads(line))

            correct = 0
            incorrect = 0
            for i in range(len(data)):
                knowledge = data[i]["knowledge"]
                question = data[i]["question"]
                hallucinated_answer = data[i]["hallucinated_answer"]
                right_answer = data[i]["right_answer"]

                if random.random() > 0.5:
                    answer = hallucinated_answer
                    ground_truth = "Yes"
                else:
                    answer = right_answer
                    ground_truth = "No"

                ans = self.get_qa_response(model, question, answer, instruction)
                ans = ans.replace(".", "")

                if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                    gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!"}
                    self.dump_jsonl(gen, output_path, append=True)
                    incorrect += 1
                    print('sample {} fails......'.format(i))
                    continue
                elif "Yes" in ans:
                    if ans != "Yes":
                        ans = "Yes"
                    gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
                elif "No" in ans:
                    if ans != "No":
                        ans = "No"
                    gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
                else:
                    gen = None
                    incorrect += 1

                assert(gen is not None)

                if ground_truth == ans:
                    correct += 1
                else:
                    incorrect += 1

                print('sample {} success......'.format(i))
                self.dump_jsonl(gen, output_path, append=True)

            # print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct/len(data)))
        return correct, incorrect


    def evaluation_dialogue_dataset(self, model, file, instruction, output_path):
        with open(file, 'r', encoding="utf-8") as f:
            data = []
            for line in f:
                data.append(json.loads(line))

            correct = 0
            incorrect = 0
            for i in range(len(data)):
                knowledge = data[i]["knowledge"]
                dialog = data[i]["dialogue_history"]
                hallucinated_response = data[i]["hallucinated_response"]
                right_response = data[i]["right_response"]

                if random.random() > 0.5:
                    response = hallucinated_response
                    ground_truth = "Yes"
                else:
                    response = right_response
                    ground_truth = "No"

                ans = self.get_dialogue_response(model, dialog, response, instruction)
                ans = ans.replace(".", "")

                if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                    gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!"}
                    self.dump_jsonl(gen, output_path, append=True)
                    incorrect += 1
                    print('sample {} fails......'.format(i))
                    continue
                elif "Yes" in ans:
                    if ans != "Yes":
                        ans = "Yes"
                    gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
                elif "No" in ans:
                    if ans != "No":
                        ans = "No"
                    gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
                else:
                    gen = None
                assert (gen is not None)

                if ground_truth == ans:
                    correct += 1
                else:
                    incorrect += 1

                print('sample {} success......'.format(i))
                self.dump_jsonl(gen, output_path, append=True)

            # print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))
        
        return correct, incorrect


    def evaluation_summarization_dataset(self, model, file, instruction, output_path):
        with open(file, 'r', encoding="utf-8") as f:
            data = []
            for line in f:
                data.append(json.loads(line))

            correct = 0
            incorrect = 0
            for i in range(len(data)):

                document = data[i]["document"]
                hallucinated_summary = data[i]["hallucinated_summary"]
                right_summary = data[i]["right_summary"]

                if random.random() > 0.5:
                    summary = hallucinated_summary
                    ground_truth = "Yes"
                else:
                    summary = right_summary
                    ground_truth = "No"

                ans = self.get_summarization_response(model, document, summary, instruction)
                ans = ans.replace(".", "")

                if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                    gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": "failed!"}
                    self.dump_jsonl(gen, output_path, append=True)
                    incorrect += 1
                    print('sample {} fails......'.format(i))
                    continue
                elif "Yes" in ans:
                    if ans != "Yes":
                        ans = "Yes"
                    gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
                elif "No" in ans:
                    if ans != "No":
                        ans = "No"
                    gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
                else:
                    gen = None
                assert (gen is not None)

                if ground_truth == ans:
                    correct += 1
                else:
                    incorrect += 1

                print('sample {} success......'.format(i))
                self.dump_jsonl(gen, output_path, append=True)

            # print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))
        
        return correct, incorrect


    def dump_jsonl(self, data, output_path, append=False):
        """
        Write list of objects to a JSON lines file.
        """
        mode = 'a+' if append else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
                json_record = json.dumps(data, ensure_ascii=False)
                f.write(json_record + '\n')
    
    def evaluate(self, task : str, use_rag : bool, output_filepath : str):
        """
        Evaluate a model using HaluEval.
        
        Args:
            task ('qa' | 'dialogue' | 'summarisation'): Which evaluation task you want to perform for the model.
            use_rag (bool): Whether to use RAG for the evaluation or not.
            output_filepath (str): Where to save the evaluation results.
        
        Returns:
            results (tuple[int,int]) : (correct, incorrect) How many samples the LLM got correct and incorrect from the dataset.
        """

        model = 'rag' if use_rag else 'llm'

        data = f"../halueval/data/{task}_data.json"

        instruction_file = f"../halueval/instructions/{task}/{task}_evaluation_instruction.txt"
        with open(instruction_file, 'r', encoding='utf-8') as f:
            instruction = f.read()
        
        if task == "qa":
            correct, incorrect = self.evaluation_qa_dataset(model, data, instruction, output_filepath)
        elif task == "dialogue":
            correct, incorrect = self.evaluation_dialogue_dataset(model, data, instruction, output_filepath)
        elif task == "summarization":
            correct, incorrect = self.evaluation_summarization_dataset(model, data, instruction, output_filepath)
        else:
            raise ValueError("The task must be qa, dialogue, or summarization!")

        print(f"{correct} correct samples, {incorrect} incorrect samples, Accuracy: {correct / len(data)}")

        return correct, incorrect

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HaluEval hallucination evaluation")

    parser.add_argument("--kbpath", default="context/knowledge_base", help="Directory for the RAG's knowledge base")
    parser.add_argument("--task", default="qa", help="HaluEval evaluation task to perform. Can be qa, dialogue, or summarization.")
    parser.add_argument('--rag', action="store_true", help='If True, the RAG model is evaluated. If false, the base LLM is evaluated.')
    parser.add_argument('--quantized', action="store_true", help='If True, the LLM is quantized. This is required if running on local hardware and not the supercomputer.')
    parser.add_argument('--output', default="halueval_output", help="Which folder to store the HaluEval results in")
    args = parser.parse_args()

    # Create directory for output results if they don't exist yet.
    os.makedirs(args.output, exist_ok=True)

    output_file = os.path.join(args.output, f"{args.task}_{'rag' if args.rag == True else 'llm'}_results.json")

    eval = HaluEval(kb_path=args.kbpath, quantized=args.quantized)
    correct, incorrect = eval.evaluate(task=args.task, use_rag = args.rag, output_filepath=output_file)