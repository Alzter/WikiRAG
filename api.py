from typing import Annotated # Better type hints library for Python (read: less bad)

from fastapi import FastAPI, HTTPException, File, UploadFile

from processor.wikipedia_corpus_download import WikipediaDownload

app = FastAPI()


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

@app.get("/download_wikipedia_dump/{dump_url}")
async def download_wiki_dump(dump_url : str = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2', output_dir : str = "wikipedia", download_subset:bool = False):
    """
    Download a Wikipedia dump, convert it to raw text, and save it to ``output_dir``.
    """

    output_dir = WikipediaDownload.download_and_extract_wikipedia_dump(output_dir=output_dir, dump_url=dump_url, download_subset=download_subset)

    return {
        "dump_save_path" : output_dir
    }

