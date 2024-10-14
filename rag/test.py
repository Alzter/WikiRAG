from retrieval import Retrieval
retrieval = Retrieval(".\context\knowledge_base")

#hnsw
retrieval.get_context("What is the first principle of animation?", num_contexts=1, hnsw = True,verbose=True)

#parse_retrieval
# retrieval.get_context("What is the first principle of animation?", num_contexts=1,verbose=True)