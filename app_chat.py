import gradio as gr
# from rag.iterative_retrieval import IterativeRetrieval
import os

# KB_PATH = "context/sonic" # TOOD: Add a selector interface to create this.
# rag = IterativeRetrieval(KB_PATH) # Create the RAG model.

# def chat(message, history):
#     message_text = message['text']
#     history.append({'role':'user', 'content':message_text})
#     history, response = rag.qd.generate_response(history, max_new_tokens=100)
#     return response

# def chat_rag(message, history):
#     message_text = message['text']
#     response, history, contexts = rag.answer_multi_hop_question(message_text, max_new_tokens=100)
#     return response

def chat(message, history):
    return "Dummy response"

with gr.Blocks(
    title='RAG',
    css="#col { height: 90vh !important; }",
    fill_height=True
) as ui:

    with gr.Row(equal_height=False):
        # Settings Panel
        with gr.Column(scale=3) as settings_ui:
            with gr.Blocks():
                upload_btn = gr.UploadButton(label="Upload Context as PDF")

        # Chat Panel
        with gr.Column(scale=7, elem_id='col'):
            _ = gr.ChatInterface(
                fn = chat,
                multimodal=False,
                type='messages',
                fill_height=True
            )

ui.launch()