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
    # Source: https://github.com/zylon-ai/private-gpt/issues/1377
    title="RAG system",
    css=".contain { display: flex !important; flex-direction: column !important; }"
    "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
    "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
    "#col { height: 100vh !important; }"
) as ui:

    with gr.Row(equal_height=True):
        
        # Settings Panel
        with gr.Column(scale=3) as settings_ui:
            with gr.Blocks():
                upload_btn = gr.UploadButton(label="Upload Context as PDF")

        # Chat Panel
        with gr.Column(scale=7):
            _ = gr.ChatInterface(
                fn = chat,
                multimodal=False,
                type='messages',
                fill_height=True
            )

ui.launch()