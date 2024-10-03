import numpy as np
import json, os
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from language_model import LanguageModel

class CorpusEmbedding(LanguageModel):

    def __init__(self, model_name = "jinaai/jina-embeddings-v2-base-en", quantized = False):
        return super().__init__(model_name, quantized)

    def chunk_by_sentences(raw_text_corpus_path: str, tokenizer: callable):
        """
        Split the input text into sentences using the tokenizer.
        
        Args:
            raw_text_corpus_path (str): The path of the raw text corpus to split into sentences.
            param tokenizer: The tokenizer to use.
        
        Returns:
            chunks (list): The list of text chunks.
            span_annotations (list): The location for each text chunk within the corpus.
        """

        inputs = tokenizer(raw_text_corpus_path, return_tensors='pt', return_offsets_mapping=True)
        punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
        sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
        token_offsets = inputs['offset_mapping'][0]
        token_ids = inputs['input_ids'][0]
        chunk_positions = [
            (i, int(start + 1))
            for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
            if token_id == punctuation_mark_id
            and (
                token_offsets[i + 1][0] - token_offsets[i][1] > 0
                or token_ids[i + 1] == sep_id
            )
        ]
        chunks = [
            raw_text_corpus_path[x[1] : y[1]]
            for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        span_annotations = [
            (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        return chunks, span_annotations

    def read_input_texts_from_folder(self, raw_text_corpus_path):
        """
        Read text content from all JSON files in a folder and concatenates it into a single string.
        
        Args:
            raw_text_corpus_path (str): The path of the raw text corpus to read.
        
        Returns:
            all_text (str): All text from the folder concatenated using line breaks.
        """

        all_text = ""
        for root, _, files in os.walk(raw_text_corpus_path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            # Parse each line as JSON and extract the 'text' field
                            data = json.loads(line)
                            text_content = data.get('text', '').strip()  # Strip any leading/trailing whitespace
                            if text_content:  # Ensure only non-empty content is added
                                all_text += text_content + "\n"
                        except json.JSONDecodeError:
                            continue  # Skip lines that are not valid JSON
        return all_text

    def late_chunking(self, model_output: "BatchEncoding", span_annotation: list, max_length=None):
        """
        Performs late chunking on a list of embeddings.

        Args:
            model_output (list): A list of embeddings to chunk.
            span_annotation (list): A list of chunk locations to use where each location is a tuple.
            max_length (int, optional): The maximum permitted length of chunks.
        
        Returns:
            outputs (list): The embeddings pooled into chunks.
        """
        token_embeddings = model_output[0]
        outputs = []
        for embeddings, annotations in zip(token_embeddings, span_annotation):
            if (
                max_length is not None
            ):  # remove annotations which go bejond the max-length of the model
                annotations = [
                    (start, min(end, max_length - 1))
                    for (start, end) in annotations
                    if start < (max_length - 1)
                ]
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / (end - start)
                for start, end in annotations
                if (end - start) >= 1
            ]
            pooled_embeddings = [
                embedding.detach().cpu().numpy() for embedding in pooled_embeddings
            ]
            outputs.append(pooled_embeddings)

        return outputs

    def corpus_to_embeddings(self, raw_text_corpus_path : str, output_dir : str, use_late_chunking = True):
        """
        Converts a raw text knowledge corpus into a NumPy array of chunked embeddings and saves the resulting array to ``output_dir``.

        Args:
            raw_text_corpus_path (str): The path of the raw text corpus to read.
            output_dir (str): The directory where the embeddings will be saved.
            use_late_chunking (bool, optional): If true, performs text embedding *before* text chunking. This is a technique known as [late chunking](https://arxiv.org/abs/2409.04701), which significantly improves semantic meaning of embeddings.
        
        Returns:
            output_dir (str): The directory where the embeddings were saved.
        """

        # Read the entire corpus as a single string.
        input_text = self.read_input_texts_from_folder(raw_text_corpus_path)

        chunks, span_annotations = self.chunk_by_sentences(input_text, self.tokenizer)

        # Print the combined extracted text (for debugging purposes)
        # if input_text.strip():
        #     print(f"Extracted Text: {input_text[:1000]}...")  # Print first 1000 characters to check
        # else:
        #     print("No text extracted")

        if use_late_chunking:
            inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
            model_output = self.model(**inputs)
            embeddings = self.late_chunking(model_output, [span_annotations])[0]
        else:
            embeddings = self.model.encode(chunks)
        
        print(f"Embedding successful. Saving to path: {output_dir}")
        np.save(output_dir, embeddings, allow_pickle=True)

        return output_dir
