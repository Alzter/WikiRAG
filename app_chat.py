import gradio as gr
import os, requests
from io import BytesIO
from rag.pdf_embedding import PDFEmbedding

# Configuration
KB_PATH = 'context'
WIKIPEDIA_DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

# Get the size of the latest English Wikipedia dump in megabytes.
wikipedia_dump_size_mb = None
try:
    metadata = requests.head(WIKIPEDIA_DUMP_URL)
    wikipedia_dump_size_mb = metadata.headers["content-length"]
    wikipedia_dump_size_mb = int(wikipedia_dump_size_mb)
    wikipedia_dump_size_mb // 1_000_000
except Exception: pass

# ====================================================================
# Utility Methods

def get_knowledge_bases(path = KB_PATH):
    return os.listdir(path)

def get_kb_path(kb_name):
    return os.path.join(KB_PATH, kb_name)

def get_num_contexts_for_kb(kb_name):
    return len(
        os.listdir(get_kb_path(kb_name))
    )

import re
def sanitise_string(input_string : str):
        """
        Sanitises a string to make it usable as a folder name
        by removing all non alphanumeric and whitespace characters.
        """
        return re.sub(r'[^a-zA-Z0-9|\s]', '', input_string).strip()

def select_kb(name):
    if not os.path.exists(get_kb_path(name)): raise FileNotFoundError(f"Knowledge base {name} not found at path {get_kb_path(name)}")

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


from rag.wikipedia_corpus_download import WikipediaDownload
from rag.wiki_corpus_embedding import WikiCorpusEmbedding
import shutil
wikipedia_embedding_model = WikiCorpusEmbedding()

def download_wikipedia_to_kb(
        wiki_subset_size_mb:int,
        knowledge_base_name : str,
        dump_url : str = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2',
        cache_dir : str = "cache/wikipedia_raw",
        subfile_max_size_megabytes : int = 10,
        embedding_batch_size_mb : int = 50
    ):

    output_dir = get_kb_path(knowledge_base_name)

    """
    Download a Wikipedia dump of size ``wiki_subset_size_mb``, convert it to raw text, embed it, and store it into the knowledge base.
    """

    gr.Info(f"Downloading {wiki_subset_size_mb}MB of Wikipedia to Knowledge Base {knowledge_base_name}, please wait...")

    save_path = WikipediaDownload.download_and_extract_wikipedia_dump(output_dir=cache_dir, subfile_max_megabytes = subfile_max_size_megabytes, max_megabytes=wiki_subset_size_mb, dump_url=dump_url)

    # Load the embedding and tokenizer model

    save_path = wikipedia_embedding_model.embed_wikipedia_raw_text(cache_dir, output_dir=output_dir, batch_size_mb=embedding_batch_size_mb)

    shutil.rmtree(cache_dir)

    return save_path

upload_pdf = PDFEmbedding()

def upload_pdf_to_kb(pdf_file, knowledge_base):
    filepath = pdf_file.name
    file = open(pdf_file.name, 'rb')
    filename = os.path.splitext(os.path.split(filepath)[-1])[0] # Get only the name of the file w/o extensions.
    
    output_dir = os.path.join(KB_PATH, knowledge_base)

    gr.Info(f"Uploading PDF {filename} to Knowledge Base {knowledge_base}")
    
    # Add the PDF to the knowledge base
    upload_pdf.embed_pdf_file(file, filename, output_dir=output_dir)

def pdf_success(pdf_name, knowledge_base):
    filename = os.path.splitext(os.path.split(pdf_name)[-1])[0] # Get only the name of the file w/o extensions.
    gr.Info(f"Successfully added PDF {filename} to Knowledge Base {knowledge_base}")

    return select_kb(knowledge_base)

# from rag.iterative_retrieval import IterativeRetrieval
# KB_PATH = "context/sonic" # TOOD: Add a selector interface to create this.
# rag = IterativeRetrieval(KB_PATH) # Create the RAG model.

def chat(message, history, max_tokens, use_rag, rag_max_sub_qs, rag_max_articles, rag_num_chunks):
    gr.Info(str(use_rag))

    max_tokens = int(round(max_tokens))

    # max_new_tokens : int = 100, maximum_reasoning_steps : int = 5, max_sub_question_answer_attempts : int = 1, num_chunks : int = 1

    rag_params = {
        "maximum_reasoning_steps" : rag_max_sub_qs,
        "max_sub_question_answer_attempts" : rag_max_articles,
        "num_chunks" : rag_num_chunks

    }

    return str(rag_params)

    # chat_fn = chat_rag if use_rag else chat_no_rag
    # return chat_fn(message, history, max_tokens, **rag_params)

# def chat_no_rag(message, history, max_tokens):
#     message_text = message['text']
#     history.append({'role':'user', 'content':message_text})
#     history, response = rag.qd.generate_response(history, max_new_tokens=100)
#     return response

# def chat_rag(message, history, max_tekens, maximum_reasoning_steps, max_sub_question_answer_attempts, num_chunks):
#     message_text = message['text']
#     response, history, contexts = rag.answer_multi_hop_question(
#         message_text,
#         max_new_tokens=100
#         maximum_reasoning_steps=maximum_reasoning_steps,
#         max_sub_question_answer_attempts=max_sub_question_answer_attempts,
#         num_chunks=num_chunks
#         )
#     return response
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
            
            max_tokens = gr.Slider(10, 1000, value=100, label="Max Tokens")
            
            use_rag = gr.Checkbox(label="Use RAG", value=False)

            with gr.Accordion("RAG Settings", open=False):
                
                max_sub_qs = gr.Number(value=5, minimum=1, maximum=10, label="Maximum Sub Questions")
               
                max_articles = gr.Number(value=2, minimum=1, maximum=10, label="Maximum Answer Attempts Per Sub-Question")

                max_paragraphs = gr.Number(value=3, minimum=1, maximum=10, label="Maximum Paragraphs Per Sub-Question")

                
            with gr.Accordion("Knowledge Base Settings", open=False):
                kb_selector = gr.Dropdown(get_knowledge_bases(), label="Knowledge Base")

                kb_article_count = gr.Textbox(f"{get_num_contexts_for_kb(kb_selector.value)} articles", label="Number of articles:")
                kb_selector.change(select_kb, kb_selector, kb_article_count)
            
                upload_btn = gr.UploadButton(label="Upload Context as PDF", file_types=['.pdf'])
                upload_btn.upload(upload_pdf_to_kb, inputs = [upload_btn, kb_selector]).success(pdf_success, inputs=[upload_btn, kb_selector], outputs=[kb_article_count])

                with gr.Accordion("Download Wikipedia to Knowledge Base"):
                    wiki_subset_mb = gr.Slider(5, wikipedia_dump_size_mb,  label="Wikipedia Data To Download (mb)")

                    download_btn = gr.Button("Download Wikipedia to Knowledge Base")

                    download_btn.click(download_wikipedia_to_kb, inputs = [wiki_subset_mb, kb_selector]).success(
                        lambda mb, kb: gr.Info(f"Successfully downloaded {mb}MB of Wikipedia to Knowledge Base {kb}."), inputs=[wiki_subset_mb, kb_selector]
                    )
            
            with gr.Accordion("Create new Knowledge Base", open=False):
                new_kb_name = gr.Textbox(label="Name")
                
                new_kb_create_button = gr.Button("Create")
                new_kb_create_button.click(create_kb, new_kb_name).success(update_kb_selector, inputs=new_kb_name, outputs=kb_selector)

        # Chat Panel
        with gr.Column(scale=7, elem_id='col'):
            _ = gr.ChatInterface(
                fn = chat,
                additional_inputs=[max_tokens, use_rag, max_sub_qs, max_articles, max_paragraphs],
                multimodal=False,
                type='messages',
                fill_height=True
            )

ui.launch(show_error=True)