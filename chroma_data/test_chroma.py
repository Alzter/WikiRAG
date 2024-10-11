import chromadb
import chromadb.utils.embedding_functions as embedding_functions
# client = chromadb.HttpClient()
client = chromadb.PersistentClient(path="/chroma_data/data")

jinaai_ef = embedding_functions.JinaEmbeddingFunction(
                api_key="jina_f035ef11ca0e4ea783e89903692702f9DOElWbS2PX630hSXuhH6TYcPvvN3",
                model_name="jina-embeddings-v2-base-en"
            )
jinaai_ef(input=["This is my first text to embed", "This is my second document"])

collection = client.create_collection("sample_collection", embedding_function=jinaai_ef)


# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=["This is document1", "This is document2"], # we embed for you, or bring your own
    metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on arbitrary metadata!
    ids=["doc1", "doc2"], # must be unique for each doc
)
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)
# 


