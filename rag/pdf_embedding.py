import sys; sys.path.append("processor")
from document_embedding import DocumentEmbedding

from typing import Union, IO, Any
from pathlib import Path
StreamType = IO[Any]
StrByteType = Union[str, StreamType]

import re
from re import _FlagsType
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

class PDFCorpusEmbedding(DocumentEmbedding):

    def __init__(self, fast : bool = False):
        """
        Args:
            fast (bool): If True, quantizes the embedding model. This leads to faster embedding time, but worse embeddings.
        """

        return super().__init__(fast=fast)

    def extract_pdf_text(self, file : StrByteType | Path, parsers : list[tuple[str,str,_FlagsType | None]] = default_parsers):
        """
        Extract all text from an uploaded PDF file and return as a string.

        Args:
            file (StrByteType | Path): The file given either as a string path string or a stream of bytes.
            parsers (list[tuple[string, string, _FlagsType | None]], optional):
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

    def pdf_to_embedding(self, file : StrByteType | Path, output_dir : str):
        """
        Convert a PDF into embeddings and save it into the knowledge base in ``output_dir``.
        """

        paragraphs = self.extract_pdf_text(file)

