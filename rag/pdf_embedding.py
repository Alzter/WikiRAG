import numpy as np
import json, os
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from glob import glob

import sys; sys.path.append("processor")
from document_embedding import DocumentEmbedding

import unicodedata
import re
from tqdm import tqdm
import math

class PDFCorpusEmbedding(DocumentEmbedding):

    def extract_raw_text