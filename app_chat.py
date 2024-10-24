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

# ====================================================================
# State variables
state = {}
def set_state(variable, value):
    state[variable] = value
    print(state)
def get_state(variable): return state.get(variable) if variable in state.keys() else None
# ====================================================================

# ====================================================================
# Utility Methods

def upload_pdf_to_kb(pdf_file, event: gr.EventData):
    file = open(pdf_file.name, 'rb')
    print(file)

def chat(message, history):
    return "Dummy response"
# ====================================================================

# UI

with gr.Blocks(
    title='RAG',
    css="#col { height: 90vh !important; }",
    fill_height=True
) as ui:

    with gr.Row(equal_height=False):

        # Settings Panel
        with gr.Column(scale=3, elem_id='col') as settings_ui:
            
            max_tokens = gr.Slider(10, 1000, label="Max Tokens")
            max_tokens.change(lambda value : set_state("max_tokens", value), max_tokens)

            use_rag = gr.Checkbox(label="Use RAG")
            use_rag.select(lambda value : set_state("rag_enabled", value), use_rag)

            upload_btn = gr.UploadButton(label="Upload Context as PDF", file_types=['.pdf'])
            upload_btn.upload(upload_pdf_to_kb, upload_btn)

            with gr.Accordion("RAG Settings"):
                
                kb_path = gr.Textbox(label="Knowledge Base Path", info="Path which contains your Knowledge Bases")
                kb_path.change(lambda value : set_state("kb_path", value), kb_path)
                pass

        # Chat Panel
        with gr.Column(scale=7, elem_id='col'):
            _ = gr.ChatInterface(
                fn = chat,
                multimodal=False,
                type='messages',
                fill_height=True
            )

ui.launch()