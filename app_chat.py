import gradio as gr
# from rag.iterative_retrieval import IterativeRetrieval
import os, requests

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

kb_path = 'context'
wikipedia_dump_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

wikipedia_dump_size_mb = None
try:
    metadata = requests.head(wikipedia_dump_url)
    wikipedia_dump_size_mb = metadata.headers["content-length"]
    wikipedia_dump_size_mb = int(wikipedia_dump_size_mb)
    wikipedia_dump_size_mb // 1_000_000
except Exception: pass

# ====================================================================
# State variables
state = {
    'knowledge_bases' : os.listdir(kb_path)
}

print(os.listdir(kb_path))

def set_state(variable, value):
    state[variable] = value
    print(state)
def get_state(variable): return state.get(variable) if variable in state.keys() else None
# ====================================================================

# ====================================================================
# Utility Methods

def create_kb(name):
    print(f"NEW KB {name}")

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


            with gr.Accordion("RAG Settings"):
                
                max_sub_qs = gr.Number(value=5, minimum=1, maximum=10, label="Maximum Sub Questions")
                max_sub_qs.change(lambda value : set_state("rag_max_sub_qs", int(value)), max_sub_qs)

               
                max_articles = gr.Number(value=5, minimum=1, maximum=10, label="Maximum Answer Attempts Per Sub-Question")
                max_articles.change(lambda value : set_state("rag_max_articles", int(value)), max_articles)

                max_paragraphs = gr.Number(value=5, minimum=1, maximum=10, label="Maximum Paragraphs Per Sub-Question")
                max_paragraphs.change(lambda value : set_state("rag_num_chunks", int(value)), max_paragraphs)

                kb_selector = gr.Dropdown(get_state("knowledge_bases"), label="Knowledge Base", info="Select a Knowledge Base")
                
                # with gr.Accordion("Create New Knowledge Base"):
                #     new_kb_name = gr.Textbox(label="Name")
                    
                #     new_kb_create_button = gr.Button("Create Knowledge Base")
                #     new_kb_create_button.click(create_kb, new_kb_name)
                
                # upload_btn = gr.UploadButton(label="Upload Context as PDF", file_types=['.pdf'])
                # upload_btn.upload(upload_pdf_to_kb, upload_btn)

                # with gr.Accordion("Download Wikipedia to Knowledge Base"):
                #     wiki_subset_mb = gr.Slider(0, wikipedia_dump_size_mb, label="Megabytes To Download")

                #     download_btn = gr.Button("Download Wikipedia to Knowledge Base")

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