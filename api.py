import numpy as np
from typing import Annotated # Better type hints library for Python (read: less bad)

from fastapi import FastAPI, HTTPException, File, UploadFile

from processor.wikipedia_corpus_download import WikipediaDownload
from processor.corpus_embedding import CorpusEmbedding

app = FastAPI()

@app.get("/download_wikipedia_dump/")
async def download_wikipedia_dump(dump_url : str = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2', output_dir : str = "context/wikipedia", download_subset:bool = False):
    """
    Download a Wikipedia dump, convert it to raw text, and save it to ``output_dir``.
    """

    save_path = WikipediaDownload.download_and_extract_wikipedia_dump(output_dir=output_dir, dump_url=dump_url, download_subset=download_subset)

    return {
        "dump_save_path" : save_path
    }

@app.get("/raw_text_corpus_to_embeddings/")
async def raw_text_corpus_to_embeddings(corpus_path : str = "context/wikipedia", output_dir : str = "context/embeddings", embedding_model : str = "jinaai/jina-embeddings-v2-base-en", use_model_quantization : bool = False, use_late_chunking : bool = True):
    """Converts a raw text knowledge corpus into a NumPy array of chunked embeddings and saves the resulting array to ``output_dir``."""
    
    # Load the embedding and tokenizer model
    model = CorpusEmbedding()

    save_path = model.corpus_to_embeddings(corpus_path, output_dir, use_late_chunking=use_late_chunking)

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