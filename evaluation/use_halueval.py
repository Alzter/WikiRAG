from evaluation.halueval import get_qa_response, get_dialogue_response, get_summarization_response
import json

def evaluate_rag_hallucination(rag_response_file, llm_response_file, model="gpt-3.5-turbo"):
    """
    :inputs:
    rag_response_file=RAG generated responses in JSON format including fields of question, answer, dialog, document
    llm_response_file=default LLM generated response used as a baseline for comparing hallucination rates
    model="gpt-3.5-turbo" from halueval.py L34
    """
    with open(rag_response_file, 'r') as rag_file, open(llm_response_file, 'r') as baseline_file:
        rag_data = json.load(rag_file)
        baseline_data = json.load(baseline_file)

        rag_hallucinations = 0
        baseline_hallucinations = 0
        total_samples = len(rag_data)

        for i in range(total_samples):
            question = rag_data[i]['question']
            dialog = rag_data[i].get('dialog', '')
            document = rag_data[i].get('document', '')
            instruction = "Evaluate hallucination in RAG response."

            # RAG QA Evaluation
            rag_answer = rag_data[i]['answer']
            rag_judgement = get_qa_response(model, question, rag_answer, instruction)
            if rag_judgement.strip().lower() == "yes":
                rag_hallucinations += 1
            
            # Baseline QA Evaluation
            baseline_answer = baseline_data[i]['answer']
            baseline_judgement = get_qa_response(model, question, baseline_answer, instruction)
            if baseline_judgement.strip().lower() == "yes":
                baseline_hallucinations += 1

            # RAG Dialogue Evaluation
            if dialog:
                rag_dialogue_judgement = get_dialogue_response(model, dialog, rag_answer, instruction)
                if rag_dialogue_judgement.strip().lower() == "yes":
                    rag_hallucinations += 1
                
                baseline_dialogue_judgement = get_dialogue_response(model, dialog, baseline_answer, instruction)
                if baseline_dialogue_judgement.strip().lower() == "yes":
                    baseline_hallucinations += 1

            # RAG Summarization Evaluation
            if document:
                rag_summary_judgement = get_summarization_response(model, document, rag_answer, instruction)
                if rag_summary_judgement.strip().lower() == "yes":
                    rag_hallucinations += 1
                
                baseline_summary_judgement = get_summarization_response(model, document, baseline_answer, instruction)
                if baseline_summary_judgement.strip().lower() == "yes":
                    baseline_hallucinations += 1

        # Calculating hallucination rates
        rag_rate = rag_hallucinations / total_samples
        baseline_rate = baseline_hallucinations / total_samples
        improvement = baseline_rate - rag_rate

        print(f"RAG Hallucination Rate: {rag_rate:.2%}")
        print(f"Baseline Hallucination Rate: {baseline_rate:.2%}")
        print(f"Reduction in Hallucination Rate: {improvement:.2%}")