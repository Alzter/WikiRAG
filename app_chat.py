import gradio as gr
from rag.iterative_retrieval import IterativeRetrieval
import os

KB_PATH = "context/sonic" # TOOD: Add a selector interface to create this.
rag = IterativeRetrieval(KB_PATH) # Create the RAG model.

def chat(message, history):
    message_text = message['text']
    history.append({'role':'user', 'content':message_text})
    history, response = rag.qd.generate_response(history, max_new_tokens=100)
    return response

def chat_rag(message, history):
    message_text = message['text']
    response, history, contexts = rag.answer_multi_hop_question(message_text, max_new_tokens=100)
    return response

chat_ui = gr.ChatInterface(
   fn = chat,
   multimodal=True,
   type='messages'
)

chat_ui.launch()