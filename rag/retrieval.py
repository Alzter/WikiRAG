import glob, os
import numpy as np

class Document():
    def __init__(self, title : str, summary : str):
        self.title = title; self.summary = summary

class Embedding():
    def __init__(self, text : str, vector):
        self.text = text; self.vector = vector


class Retrieval():
    
    def __init__(self, corpus_path : str):
        self.documents = self.get_document_summaries(corpus_path)
        self.corpus_path = corpus_path

    def get_document_summaries(self, corpus_path : str):

        """
        Given a corpus of documents, extract the document summaries for all documents into a data structure.
        """

        # Get all summary files in the corpus.
        summary_files = glob.glob(f"{corpus_path}/*/summary.txt")

        documents = []
        
        for summary_file in summary_files:
            
            # Get the parent directory of the summary file
            parent_directory = os.path.split(os.path.dirname(summary_file))[-1]

            with open(summary_file, "r") as f:
                summary_text = f.read()

            document = Document(title=parent_directory, summary=summary_text)

            documents.append(document)
    
        return documents

    def get_document_embeddings(self, document_name):
        document_embedding_path = os.path.join(self.corpus_path, document_name.strip())

        embeddings = []

        embedding_files = glob.glob(document_embedding_path + "//chunk_*.npy")
        embedding_texts = glob.glob(document_embedding_path + "//chunk_*.txt")
        
        assert(len(embedding_files) == len(embedding_texts), "Embedding raw text files should directly map to embedding data files.")

        for embedding_file, raw_text_file in zip(embedding_files, embedding_texts):
            embedding_data = np.load(embedding_file, encoding='bytes', allow_pickle=True)

            with open(raw_text_file, "r") as f:
                raw_text = f.read()
            
            embedding = Embedding(text=raw_text, vector=embedding_data)

            embeddings.append(embedding)
        
        return embeddings

            
            
