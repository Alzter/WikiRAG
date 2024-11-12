from rag.wiki_corpus_embedding import WikiCorpusEmbedding



def generate_knowledge_base(wikipedia_raw_text_path : str = "context/raw_text", output_dir : str = "context/knowledge_base", batch_size_mb : int = 50):
    """Converts a raw text knowledge corpus into a NumPy array of chunked embeddings and saves the resulting array to ``output_dir``.
    
    Articles are processed in batches of megabyte size ``batch_size_mb``."""
    
    # Load the embedding and tokenizer model
    model = WikiCorpusEmbedding()

    save_path = model.embed_wikipedia_raw_text(wikipedia_raw_text_path, output_dir=output_dir, batch_size_mb=batch_size_mb)

    return {
        "embeddings_save_path" : save_path
    }

generate_knowledge_base(
    wikipedia_raw_text_path= "D:/Studying/Swinburne Studying/2024-Semester_2/COS30018-Intelligent_Systems/RAG/data_backup/raw_text",
    output_dir = "D:/Studying/Swinburne Studying/2024-Semester_2/COS30018-Intelligent_Systems/RAG/data_backup/knowledge_base",
    batch_size_mb= 6
    )