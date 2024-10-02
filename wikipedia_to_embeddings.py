from corpus_embedding import CorpusEmbedding
from wikipedia_corpus_download import WikipediaDownload

wikidownload = WikipediaDownload()

WIKIPEDIA_DOWNLOAD_PATH = "wikipedia_extracted"
wikidownload.download_and_extract(True, output_dir=WIKIPEDIA_DOWNLOAD_PATH)

embeddings = CorpusEmbedding.corpus_to_embeddings(WIKIPEDIA_DOWNLOAD_PATH, use_late_chunking=True)