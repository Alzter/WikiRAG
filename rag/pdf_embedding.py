import sys; sys.path.append("rag")
from document_embedding import DocumentEmbedding

from typing import Union, IO, Any
from pathlib import Path
StreamType = IO[Any]
StrByteType = Union[str, StreamType]

import re
from tqdm import tqdm
from pypdf import PdfReader

# RegEx substitutions that are made to every page's text.
default_parsers = [
    # Maximum number of spaces between words: 1
    (r" +", " ", None),
    # Maximum number of line breaks between paragraphs: 2
    (r"\n\n+", r"\n\n", None),
    # Remove all spaces at the beginning of a line
    (r"^ *", "", re.MULTILINE),
    # Push all hanging sentences into the previous paragraph
    (r"\n+\n(?=[^\n]+\n\n)", "\n", None),
    # Force all paragraphs to contain at least 3 lines
    (r"\n+\n(?=[^\n]+\n[^\n]+\n\n)", "\n", None)
]

class PDFEmbedding(DocumentEmbedding):

    def __init__(self, fast : bool = False):
        """
        Args:
            fast (bool): If True, quantizes the embedding model. This leads to faster embedding time, but worse embeddings.
        """

        return super().__init__(fast=fast)

    def extract_pdf_text(self, file : StrByteType | Path, parsers : list[tuple[str,str,int | None]] = default_parsers):
        """
        Extract all text from an uploaded PDF file and return as a string.

        Args:
            file (StrByteType | Path): The PDF file given either as a string path string or a stream of bytes.
            parsers (list[tuple[string, string, int | None]], optional):
                A list of regular expression substitutions to use to parse the extracted PDF text into a usable format.
                Each parser is used with the ``re.sub()`` function, where the items map to the arguments ``pattern``, ``repl``, and ``flags`` respectively.
        
        Returns:
            raw_text (str): The raw text of the PDF file.
        """
        reader = PdfReader(file)

        page_text = [page.extract_text(extraction_mode='layout') for page in reader.pages]

        page_texts_clean = []

        for text in page_text:

            for pattern, repl, flags in parsers:
                if flags == None: flags = 0
                text = re.sub(pattern, repl, string=text, flags=flags)

            text = text.strip()

            page_texts_clean.append(text)
        
        page_texts_concatenated = "\n\n".join(page_texts_clean)

        return page_texts_concatenated

    def embed_pdf_file(self, file : StrByteType | Path, file_title : str, output_dir : str, overwrite : bool = True):
        """
        Convert a PDF into embeddings and save it into the knowledge base in ``output_dir``.

        Args:
            file (StrByteType | Path): The PDF file given either as a string path string or a stream of bytes.
            file_title: The name of the PDF file without the extension.
            output_dir (str): The directory to create the folder where the embeddings will be saved.
            overwrite (bool): Whether to overwrite the document embeddings if they already exist.

        Returns:
            article_path (str): The folder where the embeddings were saved.
        """

        body_text = self.extract_pdf_text(file)
        paragraphs = self.chunk_text(body_text)

        # Extremely lazy way to summarise a PDF document - get the first 30 lines.
        # TODO: Think of a better way to automatically generate summaries of PDF files.
        summary = "\n".join(body_text.splitlines()[:30])

        return self.create_document_embedding(title = file_title, summary=summary, paragraphs=paragraphs, output_dir=output_dir, overwrite=overwrite)

