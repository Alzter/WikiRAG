import gradio as gr
import os, glob, re
from io import BytesIO
from rag.pdf_embedding import PDFEmbedding

# Configuration
KB_PATH = 'context'

# ====================================================================
# Load RAG

from rag.iterative_retrieval import IterativeRetrieval
rag = IterativeRetrieval(None) # Create the RAG model.

# ====================================================================
# Utility Methods

def get_knowledge_bases(path = KB_PATH):
    return os.listdir(path)

def get_kb_path(kb_name):
    if kb_name is None: return None
    path = os.path.join(KB_PATH, kb_name)
    if not os.path.exists(path): raise FileNotFoundError(f"Knowledge base {kb_name} not found at path {path}")
    return path

def get_num_contexts_for_kb(kb_name):
    if kb_name is None: return 0

    summary_files = glob.glob(f"{get_kb_path(kb_name)}/*/summary.txt")

    return len(summary_files)

def sanitise_string(input_string : str):
        """
        Sanitises a string to make it usable as a folder name
        by removing all non alphanumeric and whitespace characters.
        """
        return re.sub(r'[^a-zA-Z0-9|\s]', '', input_string).strip()

def select_kb(name):
    if not os.path.exists(get_kb_path(name)): raise FileNotFoundError(f"Knowledge base {name} not found at path {get_kb_path(name)}")

    gr.Info(f"Loading knowledge base: {name}\nPlease wait...")

    kb_path = get_kb_path(name)

    rag.load_corpus(kb_path)

    gr.Info(f"Knowledge base {name} successfully loaded.")

    return f"{get_num_contexts_for_kb(name)} articles"

# Create a knowledge base with name 'name'
def create_kb(name):
    kb_dir = os.path.join(KB_PATH, name)

    name = sanitise_string(name)

    if name == "":
        raise ValueError("Name must be provided")
    
    if os.path.exists(kb_dir):
        raise FileExistsError(f"Knowledge base {name} already exists")
    
    os.makedirs(kb_dir)

    return name

def update_kb_selector(new_kb_name : str):
    """
    Called when user creates a new KB.
    """
    new_kb_name = sanitise_string(new_kb_name)
    gr.Info(f"Knowledge base created: {new_kb_name}")

    return gr.Dropdown(choices = get_knowledge_bases(), value=new_kb_name)

upload_pdf = PDFEmbedding()

def upload_pdf_to_kb(pdf_file, knowledge_base):
    filepath = pdf_file.name
    file = open(pdf_file.name, 'rb')
    filename = os.path.splitext(os.path.split(filepath)[-1])[0] # Get only the name of the file w/o extensions.
    
    output_dir = os.path.join(KB_PATH, knowledge_base)

    gr.Info(f"Uploading PDF {filename} to knowledge base {knowledge_base}")
    
    # Add the PDF to the knowledge base
    upload_pdf.embed_pdf_file(file, filename, output_dir=output_dir)

def pdf_success(pdf_name, knowledge_base):
    filename = os.path.splitext(os.path.split(pdf_name)[-1])[0] # Get only the name of the file w/o extensions.
    gr.Info(f"Successfully added PDF {filename} to knowledge base {knowledge_base}")

    return select_kb(knowledge_base)

def chat(message, history, max_tokens, use_rag, kb_name, rag_max_sub_qs, rag_max_articles, rag_num_chunks):

    if use_rag:
        if kb_name is None:
            raise ValueError("A knowledge base must be selected to use the RAG model.")
        
        if get_num_contexts_for_kb(kb_name) == 0:
            raise ValueError("No documents have been added to the RAG knowledge base. Add documents first before using RAG!")

    max_tokens = int(round(max_tokens))

    # max_new_tokens : int = 100, maximum_reasoning_steps : int = 5, max_sub_question_answer_attempts : int = 1, num_chunks : int = 1

    rag_params = {
        "maximum_reasoning_steps" : rag_max_sub_qs,
        "max_sub_question_answer_attempts" : rag_max_articles,
        "num_chunks" : rag_num_chunks

    }

    chat_fn = chat_rag if use_rag else chat_no_rag
    return chat_fn(message, history, max_tokens, rag_params)

def chat_no_rag(message, history, max_tokens, rag_params):
    history.append({'role':'user', 'content':message})
    history, response = rag.qd.generate_response(history, max_new_tokens=max_tokens)
    return response

def chat_rag(message, history, max_tokens, rag_params):
    response, history, contexts = rag.answer_multi_hop_question(
        message,
        max_new_tokens=max_tokens,
        **rag_params
        )
    return response

# ====================================================================
# Initialisation

# Load the first KB corpus if it exists.
kbs = get_knowledge_bases()
if len(kbs) > 0:
    first_kb = kbs[0]
    select_kb(first_kb)

# ====================================================================

# UI

with gr.Blocks(
    theme="citrus",
    title='RAG',
    css="#col { height: 90vh !important; }",
    fill_height=True
) as ui:

    with gr.Row(equal_height=False):

        # Settings Panel
        with gr.Column(scale=3) as settings_ui:
            
            max_tokens = gr.Slider(10, 1000, value=100, step=1, label="Max Tokens")
            
            use_rag = gr.Checkbox(label="Use RAG", value=False)

            with gr.Accordion("RAG Settings", open=False):
                
                max_sub_qs = gr.Slider(value=5, minimum=1, maximum=10, step=1, label="Maximum Sub-Questions", info="How many times shall the RAG be allowed to decompose the user question into smaller sub-questions?")
               
                max_articles = gr.Slider(value=2, minimum=1, maximum=10, step=1, label="Maximum Answer Attempts Per Sub-Question", info="During each answer attempt, the RAG will fetch a different document to answer the sub-question.")

                max_paragraphs = gr.Slider(value=3, minimum=1, maximum=10, step=1, label="Number of Paragraphs Per Article", info="How many paragraphs should be retrieved for each document during the sub-question answer attempt?")

                
            with gr.Accordion("Knowledge Base Settings", open=False):
                kb_selector = gr.Dropdown(get_knowledge_bases(), label="Knowledge Base")

                kb_article_count = gr.Textbox(f"{get_num_contexts_for_kb(kb_selector.value)} articles", label="Number of articles:")
                kb_selector.change(select_kb, kb_selector, kb_article_count)
            
                upload_btn = gr.UploadButton(label="Upload PDF Document", file_types=['.pdf'])
                upload_btn.upload(upload_pdf_to_kb, inputs = [upload_btn, kb_selector]).success(pdf_success, inputs=[upload_btn, kb_selector], outputs=[kb_article_count])
            
            with gr.Accordion("Create New Knowledge Base", open=False):
                new_kb_name = gr.Textbox(label="Name")
                
                new_kb_create_button = gr.Button("Create")
                new_kb_create_button.click(create_kb, new_kb_name).success(update_kb_selector, inputs=new_kb_name, outputs=kb_selector).success(select_kb, kb_selector, kb_article_count)

        # Chat Panel
        with gr.Column(scale=7, elem_id='col'):
            _ = gr.ChatInterface(
                fn = chat,
                additional_inputs=[max_tokens, use_rag, kb_selector, max_sub_qs, max_articles, max_paragraphs],
                multimodal=False,
                type='messages',
                fill_height=True
            )

ui.launch(show_error=True)