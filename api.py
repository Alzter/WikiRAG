import numpy as np
from typing import Annotated # Better type hints library for Python (read: less bad)

from fastapi import FastAPI, HTTPException, File, UploadFile

from rag.wikipedia_corpus_download import WikipediaDownload
from rag.corpus_embedding import CorpusEmbedding

app = FastAPI()

@app.get("/download_wikipedia_dump/")
async def download_wikipedia_dump(dump_url : str = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2', output_dir : str = "context/raw_text", download_subset:bool = False):
    """
    Download a Wikipedia dump, convert it to raw text, and save it to ``output_dir``.
    """

    save_path = WikipediaDownload.download_and_extract_wikipedia_dump(output_dir=output_dir, dump_url=dump_url, download_subset=download_subset)

    return {
        "dump_save_path" : save_path
    }

# @app.get("/extract_raw_text_from_wikipedia/")
# async def extract_raw_text_from_wikipedia_dump(dump_file_path : str, output_dir : str = "context/wikipedia", is_subset:bool = False):
#     """
#     Extract raw text from a Wikipedia dump using WikiExtractor and save it to ``output_dir``.
#     """

#     save_path = WikipediaDownload.extract_wikipedia_dump(dump_file_path, output_dir=output_dir, is_subset=is_subset)

#     return {
#         "extracted_save_path" : save_path
#     }

@app.get("/generate_knowledge_base/")
async def generate_knowledge_base(wikipedia_raw_text_path : str = "context/raw_text", output_dir : str = "context/knowledge_base"):
    """Converts a raw text knowledge corpus into a NumPy array of chunked embeddings and saves the resulting array to ``output_dir``."""
    
    # Load the embedding and tokenizer model
    model = CorpusEmbedding()

    save_path = model.embed_wikipedia_raw_text(wikipedia_raw_text_path, output_dir)

    return {
        "embeddings_save_path" : save_path
    }

@app.get("/query_rag/{query}")
async def query_rag(query):

    # TODO: Just draw the f--- owl
    
    raise HTTPException(501, "RAG query method not yet implemented.")
    return {"response": f"Not implemented yet: {query}"}

# @app.post("/preprocess_document")
# async def preprocess_document(document : UploadFile):

#     filename = document.filename
#     extension = document.content_type

#     # TODO: Raise assertion that documents must be of PDF type

#     raise HTTPException(501, "Document pre-process method not yet implemented.")
#     return {"message": "Not implemented yet"}