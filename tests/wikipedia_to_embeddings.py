import numpy as np
import os

from processor.wiki_corpus_embedding import WikiCorpusEmbedding
from processor.wikipedia_corpus_download import WikipediaDownload

WIKIPEDIA_DOWNLOAD_PATH = "context\\wikipedia_extracted"

out_path = input("Enter the name of the folder you wish to save the embeddings of Wikipedia to:")
os.makedirs(out_path)
download_all = bool(input("Download entire Wikipedia dump or 5MB subset?\nPress 0 for entire Wikipedia dump, 1 for 5mb subset."))

print(f"Downloading Wikipedia as raw text and saving to {WIKIPEDIA_DOWNLOAD_PATH}")
WikipediaDownload.download_and_extract_wikipedia_dump(WIKIPEDIA_DOWNLOAD_PATH, download_subset=download_all)

print(f"Embedding Wikipedia dump and saving to folder: {out_path}")
embedding_path = WikiCorpusEmbedding.embed_wikipedia_raw_text(WIKIPEDIA_DOWNLOAD_PATH, use_late_chunking=True)