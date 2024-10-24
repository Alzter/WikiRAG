import gradio as gr

# from rag.iterative_retrieval import IterativeRetrieval as rag

def chat_rag(message, history):
    print(message)
    print(history)
    return "Hi"

chat_ui = gr.ChatInterface(
    chat_rag
)

chat_ui.launch()