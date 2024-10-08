
import glob, os

class Document():
    def __init__(self, title : str, summary : str):
        self.title = title; self.summary = summary

class Embedding():
    def __init__(self, text : str, vector):
        self.text = text; self.vector = vector


class Retrieval():
    
    def __init__(self, corpus_path : str):
        self.documents = self.get_document_summaries(corpus_path)

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

            print(summary_file)

            with open(summary_file, "r") as f:
                summary_text = f.read()

            document = Document(title=parent_directory, summary=summary_text)

            documents.append(document)
    
        return documents

