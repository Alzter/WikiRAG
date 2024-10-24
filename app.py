import gradio as gr
from rag.iterative_retrieval import IterativeRetrieval


# from rag.iterative_retrieval import IterativeRetrieval as rag

def __init__(corpus_path, num_threads : int = 5):
    rag = IterativeRetrieval(corpus_path, num_threads=num_threads)


def history_to_chat_template(history : list[list], message : str = None) -> list[dict]:
    """
    Convert the simple history into a
    [Chat Template](https://huggingface.co/docs/transformers/v4.45.1/en/chat_templating).
    
    Args:
        history (list[list]): The chat history as a list of 2-item lists,
                               where each tuple is: [user message, bot message].
        message (str, optional):
                              The user's current message. This is added to the end of the chat history.
    
    Returns:
        chat_history (list[dict]): The history in chat template form.
    """

    chat_history = []
    for item in history:
        user_msg, assistant_msg = item

        chat_history.append({
            "role":"user","content":user_msg
        })
        chat_history.append({
            "role":"assistant","content":assistant_msg
        })
    
    if message is not None:
        chat_history.append({
            "role":"user","content":message
        })
    
    return chat_history

def chat(message, history):
    chat_history = history_to_chat_template(history, message=message)

    if hasattr(self, "rag"):
        rag.llm


def chat_rag(message, history):
    chat_history = history_to_chat_template(history, message=message)

    return "Hi"

chat_ui = gr.ChatInterface(
    chat_rag
)

chat_ui.launch()