import gradio as gr
#from rag.iterative_retrieval import IterativeRetrieval
import os

def get_knowledge_bases_in_dir(kb_directory):
    return os.listdir(kb_directory)

KB_PATH = 'context'
knowledge_bases = get_knowledge_bases_in_dir(KB_PATH)

state = {}

def set_state(key, value):
    state[key] = value

def get_state(key):
    return state.get(key)

with gr.Blocks() as kb_picker:
    
    gr.Markdown(
        """
        #COS30018 - RAG System

        Created by Matt, Alex, and Toan.
        """
    )

    with gr.Blocks():
        gr.Markdown("""
        ## Select or create a Knowledge Base:
        """)

        with gr.Row():
            kb_create_button = gr.Button("Create Knowledge Base")
            kb_open_button = gr.Button("Open Knowledge Base")

    kb_open_button.click(
        fn =set_state,
        inputs=["kb_select", "open"],
        outputs=None
    )

    kb_dropdown = gr.Dropdown(
        choices=knowledge_bases,
        label="Knowledge Base",
        info="Select a knowledge base to use:",
        visible = get_state("kb_select") == "open"
    )

kb_picker.launch()

#chat_ui = gr.ChatInterface(
#    chat_rag
#)

#chat_ui.launch()