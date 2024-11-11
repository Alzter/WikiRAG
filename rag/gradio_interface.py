import gradio as gr
import requests

API_url = "http://127.0.0.1:8000"

# Define the functions for interacting with FastAPI endpoints
def query_llm_gradio(query: str, type:str = "/query/"):
    url = f"{API_url}{type}{query}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "No answer returned.")
        reasoning = data.get("reasoning", "Answer provided by Llama 3.1")
        return answer, reasoning
    except requests.exceptions.RequestException as e:
        return f"Error: {e}", "No reasoning due to error."

def query_rag_gradio(query: str, type: str = "/query_rag/"):
    url = f"{API_url}{type}{query}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "No answer returned.")
        reasoning = data.get("reasoning", "No reasoning provided.")
        reason_process = ""
        for reasons in reasoning[-2]:
            for sub_reason in reasons:
                reason_process += f"\n{sub_reason['content']}\n"
        print(f"Reasoning process: {reason_process})
        return answer, reason_process
    except requests.exceptions.RequestException as e:
        return f"Error: {e}", "No reasoning due to error."

def add_pdf_to_knowledge_base_gradio(pdf_file):
    url = "http://127.0.0.1:8000/add_pdf_to_knowledge_base"
    files = {'document': (pdf_file.name, open(pdf_file.name, 'rb'), 'application/pdf')}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        data = response.json()
        return f"Document added successfully: {data.get('embeddings_save_path', 'Path not provided')}"
    except requests.exceptions.RequestException as e:
        return f"Error adding document: {e}"

# Set up the Gradio interface with query functions and document upload
with gr.Blocks(theme="gradio/monochrome") as demo:
    gr.Markdown("""
    # Welcome To Our RAG
    ## Team 5 - Option C COS30018
    ## This is Alex, Toan, and Matthew
    Start the web interface for our system
    """)
    with gr.Blocks():
        # Section for document uploading
        gr.Markdown("### Add a Document to the Knowledge Base")
        pdf_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
        upload_button = gr.Button("Upload PDF")
        upload_output = gr.Textbox(label="Upload Status")

        # Define what happens when the upload button is clicked
        upload_button.click(
            fn=add_pdf_to_knowledge_base_gradio, 
            inputs=pdf_input, 
            outputs=upload_output
        )

    with gr.Blocks():
        # Input box for the query
        query_input = gr.Textbox(label="Enter your query")

        # Two buttons to select the function
        with gr.Row():
            llm_button = gr.Button("Query LLM")
            rag_button = gr.Button("Query RAG")
        
        # Outputs for answer and reasoning
        answer_output = gr.Textbox(label="Answer")
        reasoning_output = gr.Textbox(label="Reasoning")

        # Define what happens when each button is clicked
        llm_button.click(
            fn=query_llm_gradio, 
            inputs=query_input, 
            outputs=[answer_output, reasoning_output]
        )
        rag_button.click(
            fn=query_rag_gradio, 
            inputs=query_input, 
            outputs=[answer_output, reasoning_output]
        )
    
    

# Launch the Gradio app
demo.launch()