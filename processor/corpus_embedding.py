import numpy as np
import json, os
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

import sys; sys.path.append("processor")
from language_model import LanguageModel

class CorpusEmbedding(LanguageModel):

    def __init__(self, model_name = "jinaai/jina-embeddings-v2-base-en", causal = False, quantized = False, use_gpu=True):
        return# super().__init__(model_name, causal, quantized)
    
        # Uncomment this line if you want to use the embedding / tokenizer model

    def read_input_texts_from_folder(self, raw_text_corpus_path, return_as_string):
        """
        Read text content from all JSON files in a folder and returns it as an array where each element represents a JSON entry in each file.
        Used to extract all Wikipedia articles from a raw text Wikipedia dump. Every entry of the array is a Wikipedia article.
        
        Args:
            raw_text_corpus_path (str): The path of the raw text corpus to read.
            return_as_string (bool): If true, all text is returned concatenated into a single string using line breaks.
        
        Returns:
            all_text (list | str): All text from the folder either as an array for each article or concatenated.
        """
        
        text_list = []

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
                                text_list.append(text_content)
                        except json.JSONDecodeError:
                            continue  # Skip lines that are not valid JSON
        
        if return_as_string: text_list = "\n".join(text_list)
        return text_list

    def parse_wikipedia_corpus(self, corpus):
        
        articles = []
        
        for article in corpus:
            paragraphs = article.split("\n\n")
            try:
                assert((len(paragraphs)>1), "Article must have more than 2 paragraphs.")

                title = paragraphs[0] # Title
                summary = paragraphs[1] # First paragraph of article = summary
                body_paragraphs = paragraphs[1:] # All paragraphs excluding title

                # Create data structure to represent Wikipedia article
                articles.append({
                    "title" : title.strip(),
                    "summary" : summary,
                    "paragraphs" : body_paragraphs
                })

            except Exception as e:
                print(f"Error parsing Wikipedia article. Traceback: {str(e)}\nArticle text:\n{article}")

        return articles

            

    def chunk_by_sentences(self, input_text: str):
            """
            Split the input text into sentences using the tokenizer.
            
            Args:
                input_text (str): The text snippet to split into sentences.
            
            Returns:
                chunks (list): The list of text chunks.
                span_annotations (list): The location for each text chunk within the corpus.
            """
            inputs = self.tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
            punctuation_mark_id = self.tokenizer.convert_tokens_to_ids('.')
            sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
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
                input_text[x[1] : y[1]]
                for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
            ]
            
            # print( f"chunk_position from 1: {chunk_positions[:-1]}\nFull chunk_pos: {chunk_positions}\n chunks: {chunks}")
            # for x, y in zip([(1, 0)]+ chunk_positions[:-1], chunk_positions):
            #     print(f"x0: {x[0]} - y0: {y[0]}") 
            #     print(f"x1: {x[1]} - y1: {y[1]}") 
            # print(f"test input_text {input_text[0:82]}")
            # print(f" what print out {[input_text[x[0] : y[0]] for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)]}")
            
            span_annotations = [
                (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
            ]
            
            # print(f"chunks:{chunks}\nspan_annotations{span_annotations}")
            return chunks, span_annotations

    def late_chunking(self, input_text : str, max_length=None):
        """
        Performs late chunking on an input raw text by embedding it and then chunking.

        Args:
            input_text (str): The text to convert into chunks.
            tokenizer (callable): The tokenizer to use.
            span_annotation (list): A list of chunk locations to use where each location is a tuple.
            max_length (int, optional): The maximum permitted length of chunks.
        
        Returns:
            outputs (list): The embeddings pooled into chunks.
        """

        chunks, span_annotations = self.chunk_by_sentences(input_text)
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        model_output = self.model(**inputs)

        token_embeddings = model_output[0]
        outputs = []
        for embeddings, annotations in zip(token_embeddings, span_annotations):
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

        wikipedia = self.read_input_texts_from_folder(raw_text_corpus_path, False)
        articles = self.parse_wikipedia_corpus(wikipedia)

        if not os.path.exists(output_dir): os.makedirs(output_dir)

        for article in articles:
            title = article['title']
            summary = article['summary']
            paragraphs = article['paragraphs']

            article_path = os.path.join(output_dir, title.strip())
            print(article_path)

            if not os.path.exists(article_path): os.mkdir(article_path)

            summary_file_path = os.path.join(article_path, "summary.txt")

            # Write the summary file.
            summary_file = open(summary_file_path, "wb")
            summary_file.write(summary.encode("utf-8"))
            summary_file.close()

            # Treat each paragraph as a chunk
            for paragraph_id, paragraph in enumerate(paragraphs):

                embedding_file_path = os.path.join(article_path, f"chunk_{paragraph_id}.txt")

                print(f"Embed paragraph {paragraph_id} for article {title}")

                print(embedding_file_path)

                embedding_file = open(embedding_file_path, "wb")
                embedding_file.write(paragraph.encode("utf-8"))
                embedding_file.close()

                # if use_late_chunking:
                #     embeddings = self.late_chunking(input_text)
                # else:
                #     chunks, span_annotations = self.chunk_by_sentences(input_text)
                #     embeddings = self.model.encode(chunks)
            
        #print(f"Embedding successful. Saving to path: {output_dir}")
        #np.save(output_dir, embeddings, allow_pickle=True)

        return output_dir
