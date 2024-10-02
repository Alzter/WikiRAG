import numpy as np

from corpus_embedding import CorpusEmbedding
from wikipedia_corpus_download import WikipediaDownload

wikidownload = WikipediaDownload()

WIKIPEDIA_DOWNLOAD_PATH = "context\\wikipedia_extracted"

print(f"Downloading Wikipedia as raw text and saving to {WIKIPEDIA_DOWNLOAD_PATH}")
wikidownload.download_and_extract_wikipedia_dump(WIKIPEDIA_DOWNLOAD_PATH, download_subset=True)


EMBEDDINGS_SAVE_PATH = "context\\wikipedia_embeddings"

print(f"Embedding Wikipedia dump and saving to folder: {EMBEDDINGS_SAVE_PATH}")
embeddings = CorpusEmbedding.corpus_to_embeddings(WIKIPEDIA_DOWNLOAD_PATH, use_late_chunking=True)

print("Embedding successful. Saving Wikipedia corpus.")
np.save(EMBEDDINGS_SAVE_PATH, embeddings, allow_pickle=True)