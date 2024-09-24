from typing import Annotated # Better type hints library for Python (read: less bad)

from fastapi import FastAPI, HTTPException, File, UploadFile

app = FastAPI()


@app.get("/query_rag/{query}")
async def query_rag(query):

    # TODO: Just draw the f--- owl
    
    raise HTTPException(501, "RAG query method not yet implemented.")
    return {"response": f"Not implemented yet: {query}"}

@app.post("/preprocess_document")
async def preprocess_document(document : UploadFile):

    filename = document.filename
    extension = document.content_type

    # TODO: Raise assertion that documents must be of PDF type

    raise HTTPException(501, "Document pre-process method not yet implemented.")
    return {"message": "Not implemented yet"}