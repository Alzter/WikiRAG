import glob, os
import numpy as np
from embedding_model import EmbeddingModel
import k_best
from rank_bm25 import BM25Okapi
import torch
from tqdm import tqdm
import hnswlib

class Document():
    def __init__(self, title : str, summary : str, embedding : torch.Tensor):
        self.title = title; self.summary = summary; self.embedding = embedding

class Embedding():
    def __init__(self, raw_text : str, embedding : torch.Tensor):
        self.raw_text = raw_text; self.embedding = embedding

class HNSW():
    """
    Hierarchical Navigable Small World graphs (HNSW) is an algorithm that allows for 
    efficient nearest neighbor search, and the Sentence Transformers library allows for 
    the generation of semantically meaningful sentence embeddings.
    """
    def __init__(self, 
                corpus: list[Document],
                ef: int, 
                space = 'l2',   # 'l2' refers to the Euclidean distance
                ef_construction = 200,
                M = 16,
                num_threads = 4
        ):
            
        self.corpus = corpus
        self.corpus_embeddings = [document.embedding for document in corpus]
        self.dim = corpus[0].embedding.shape[-1]
        self.ef = ef
        self.threads = num_threads

        # Build the index
        self.hnsw = None
        self.generate_hnsw(
            space, 
            num_elements=len(self.corpus_embeddings),
            ef_construction = ef_construction,
            M = M,
            data = self.corpus_embeddings
        )
        
    def generate_hnsw(self, space, num_elements, ef_construction, M, data):
        self.hnsw = hnswlib.Index(space = space, dim = self.dim)  # 'l2' refers to the Euclidean distance
        self.hnsw.set_num_threads(self.threads)
        self.hnsw.init_index(
            max_elements = num_elements, 
            ef_construction = ef_construction, 
            M = M)
        
        # Add items to HNSW index
        # for i, doc in enumerate(data):
        #     # Convert each torch tensor to NumPy array and remove batch dimension (if necessary)
        #     numpy_embedding = doc.squeeze(0).numpy()  # Remove batch dim [1, 384] -> [384]
        #     # print(numpy_embedding.shape)  # Should be (384,)
        #     # print(numpy_embedding)
        #     self.hnsw.add_items(numpy_embedding, i)  # Add to hnswlib with index 'i'
        
        for doc in data:
            self.hnsw.add_items(doc[-1])    # Add to hnswlib

    @staticmethod
    def get_scores(hnsw, query : torch.Tensor, k : int) -> list[float]:
        labels, distances = hnsw.knn_query(query, k)

        print(f"Scores: {labels}")
        print(f"Length of scores: {len(labels)}")

        return labels[0], distances

    # @staticmethod
    def get_k_best_documents(self, k : int, query : torch.Tensor) -> list[Embedding]:
        if self.ef <= k:
            self.ef = k + 1 
            self.hnsw.set_ef(self.ef)  # ef should always be greater than k
        
        scores, _ = HNSW.get_scores(self.hnsw, query, k = 10)

        print(f"Length of corpus: {len(self.corpus)}")
        top_k = k_best.get_k_best(k, self.corpus, scores)
        return top_k

class SparseRetrieval():
    """
    Sparse document retrieval technique. This class retrieves documents
    using lexical matching, i.e., the number of exact word matches.

    AKA, an N-gram retrieval method.
    """

    @staticmethod
    def get_scores(query : str, corpus : list[Document]) -> list[float]:
        """
        Given an input query string and a corpus (a list of document strings),
        evaluate the lexical similarity between each document in the corpus and the query
        on a scale from 0 to 1. Return the list of scores for each corpus document.

        Source: https://pypi.org/project/rank-bm25/

        Args:
            query (str): "windy London"
            corpus (list[Document]): ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]

        Returns:
            scores (ndarray[int]): array([0.        , 0.93729472, 0.        ])
        """

        # Extract summary text all from documents
        corpus = [document.summary for document in corpus]

        tokenized_corpus = [doc.split(" ") for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.split(" ")

        return bm25.get_scores(tokenized_query)

    @staticmethod
    def get_k_best_documents(k : int, query : str, corpus : list[Document]) -> list[Document]:
        """
        Given an input query, retrieve k most similar documents from the corpus
        using sparse retrieval (lexical matching) in descending order of similarity.

        Args:
            n (int): The number of documents to retrieve. E.g., 1.
            query (str): "windy London"
            corpus (list[Document]): ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]

        Returns:
            top_k (list[str]): The k most matching items in the corpus. E.g., ['It is quite windy in London']
        """

        scores = SparseRetrieval.get_scores(query, corpus)

        top_k = k_best.get_k_best(k, corpus, scores)

        return top_k

class DenseRetrieval():
    """
    Dense retrieval technique using document embeddings.
    This class retrieves documents by calculating their
    semantic similarity using a cosine similarity function.
    """

    @staticmethod
    def compare_documents(a : torch.Tensor, b : torch.Tensor) -> float:
        """
        Returns the similarity between embeddings a and b using cosine similarity.
        """
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    @staticmethod
    def get_scores(query : torch.Tensor, corpus : list[Embedding]) -> list[float]:
        """
        Given an input query embedding and a corpus (a list of document embeddings),
        evaluate the semantic similarity between each document in the corpus and the query
        on a scale from 0 to 1. Return the list of scores for each corpus document.
        """
        corpus_embeddings = [document.embedding for document in corpus]

        scores = []

        for embedding in corpus_embeddings:
            similarity = DenseRetrieval.compare_documents(query, embedding)[0]
            scores.append(similarity)
        
        return scores
    
    @staticmethod
    def get_k_best_documents(k : int, query : torch.Tensor, corpus : list[Embedding]) -> list[Embedding]:
        """
        Given an input query, retrieve k most similar embeddings from the corpus
        using dense retrieval (cosine similarity of embeddings) in descending order of similarity.

        Args:
            n (int): The number of documents to retrieve. E.g., 1.
            query (torch.Tensor):
            corpus (list[Embedding]):

        Returns:
            top_k (list[Embedding]): The k most matching items in the corpus.
        """

        scores = DenseRetrieval.get_scores(query, corpus)
        top_k = k_best.get_k_best(k, corpus, scores)

        return top_k

class Retrieval():
    def __init__(self, corpus_path : str, num_threads : int = 4):
        
        print("Searching for files in knowledge base...")
        
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")
        
        if len(glob.glob(corpus_path + "/*/summary.txt")) == 0:
            raise LookupError(f"No documents found in corpus. Corpus path: {corpus_path}")

        print("Loading knowledge base...")
        self.documents = self.get_document_summaries(corpus_path)
        self.corpus_path = corpus_path
        self.embedding_model = EmbeddingModel()
        
        self.hsnw_search = HNSW(self.documents, ef=50, space = 'l2', num_threads=num_threads)


    def get_document_summaries(self, corpus_path : str) -> list[Document]:

        """
        Given a corpus of documents, extract the document summaries for all documents into a data structure.
        """

        # Get all summary files in the corpus.
        summary_files = glob.glob(f"{corpus_path}/*/summary.txt")
        summary_embeddings = glob.glob(f"{corpus_path}/*/summary.npy")

        documents = []

        assert(len(summary_files) == len(summary_embeddings), "Summary text files should directly map to summary embedding files.")
        
        progress = tqdm(zip(summary_files, summary_embeddings), "Loading knowledge base", len(summary_embeddings), unit="article")

        for summary_file, summary_embedding in progress:
            
            # Get the parent directory of the summary file
            parent_directory = os.path.split(os.path.dirname(summary_file))[-1]

            # Set the loading bar's postfix to the name of the article.
            progress.set_postfix_str(parent_directory[:10] + "...")

            # Try reading the summary w/o encoding first
            try:
                with open(summary_file, "r") as f:
                    summary_text = f.read()
            except Exception:
                with open(summary_file, "r", encoding='utf-8') as f:
                    summary_text = f.read()

            embedding_data = np.load(summary_embedding, encoding='bytes', allow_pickle=True)
            embedding_data = torch.Tensor(embedding_data)

            document = Document(title=parent_directory, summary=summary_text, embedding=embedding_data)

            documents.append(document)
    
        return documents
    
    def get_document_embeddings(self, document_name) -> list[Embedding]:
        """
        If document_name == None, load *all* embeddings.
        """

        embeddings = []

        if document_name == None:
            embedding_files = glob.glob(self.corpus_path + "//*//chunk_*.npy")
            embedding_texts = glob.glob(self.corpus_path + "//*//chunk_*.txt")
        else:
            document_embedding_path = os.path.join(self.corpus_path, document_name.strip())

            embedding_files = glob.glob(document_embedding_path + "//chunk_*.npy")
            embedding_texts = glob.glob(document_embedding_path + "//chunk_*.txt")
        
        assert(len(embedding_files) == len(embedding_texts), "Embedding raw text files should directly map to embedding data files.")

        for embedding_file, raw_text_file in zip(embedding_files, embedding_texts):

            embedding_data = np.load(embedding_file, encoding='bytes', allow_pickle=True)
            embedding_data = torch.Tensor(embedding_data)

            with open(raw_text_file, "r", encoding='utf-8') as f:
                raw_text = f.read()
            
            embedding = Embedding(raw_text=raw_text, embedding=embedding_data)

            embeddings.append(embedding)
        
        return embeddings
    
    def get_context(self, query : str, num_contexts = 1, hnsw : bool = False, use_sparse_retrieval : bool = False, exhaustive : bool = False, ignored_articles : list = [], verbose = False) -> list[str]:
        """
        Given a user question, retrieve contexts in the form of paragraphs from Wikipedia articles to answer the question.

        Method:
        1. One Wikipedia article is retrieved which best answers the initial query.
        2. Paragraphs are fetched from the chosen Wikipedia article as contexts to the query.

        NOTE: This function **only works for single-hop** questions where the answer can be found from *one* article.
        For multi-hop questions, you must first decompose the question using the ``QueryDecomposition`` class.

        Args:
            query (str): The user's question. E.g., 'Who was the first person to land on the moon?'

            num_contexts (int, optional): How many paragraphs to retrieve. Defaults to 1.

            hnsw (bool, optional):
                Controls whether to use Hierarchical Navigable Small Worlds (HNSW) for retrieval to find the closest article
                to the user's query. This is an optimisation technique which makes retrieval faster.

            use_sparse_retrieval (bool, optional):
                Controls whether to use dense or sparse retrieval to find the best Wikipedia article to answer the user's query.
                This **only works** if ``hnsw`` is set to ``False``.

                If True, uses a BM25 search of article raw text summaries to find the Wikipedia article.

                If False, uses cosine similarity search of article summary embeddings to find the Wikipedia article.

            exhaustive (bool, optional):
                If True, skip the article selection step and find context by searching through **all Wikipedia articles**.

                WARNING: This has high time and computational complexity.
            
            ignored_articles (list, optional):
                A list of ignored Wikipedia articles which are excluded from context retrieval.
            
            verbose (bool, optional): If true, prints the retrieval process to the console.
        
        Returns:
            context (str | list[str]): A string or list of ``num_context`` contexts, where each context is a paragraph from a Wikipedia article.
            article (str | None): The name of the article context was retrieved from, or ``None`` if ``exhaustive`` was True.

        """
        
        if verbose: print("Embedding user query for dense retrieval:")

        # Convert query into an embedding
        query_embedding = self.embedding_model.get_embedding(query, input_is_query=True)

        if exhaustive:

            # Load all embeddings into memory (this takes a long time)
            if not hasattr(self, "all_embeddings"):

                if verbose: print("Loading all document chunk embeddings into memory...")
                self.all_embeddings = self.get_document_embeddings(None)

            embeddings = self.all_embeddings
            article = None
            # print(f"EMBEDDING: {embeddings[1].embedding.shape}") # EMBEDDING: torch.Size([1, 384])
        else:
            
            number_of_articles_to_retrieve = 1 + len(ignored_articles)

            # Retrieve a Wikipedia article for use as context for the query.
            
            if hnsw:
                if verbose: print("Finding best article to use as context with HSNW retrieval:")
                # hnsw_retrieval = HNSW(ef_Construction=200, mL = 1.5, M = 5, Mmax = 10, corpus = self.documents)
                # best_articles = hnsw_retrieval.k_nn_search(query, k = 5, ef = 50)

                best_articles = self.hsnw_search.get_k_best_documents(number_of_articles_to_retrieve, query_embedding)

            elif use_sparse_retrieval:
                if verbose: print("Finding best article to use as context with sparse retrieval:")

                # Use BM25 (sparse retrieval) to acquire one Wikipedia article
                # which has the most n-gram lexical matches to the user query.
                best_articles = SparseRetrieval.get_k_best_documents(number_of_articles_to_retrieve, query, self.documents)

            else:
                if verbose: print("Finding best article to use as context with dense retrieval:")

                # Use cosine similarity (dense retrieval) to find most semantically similar article summary to the user query.
                best_articles = DenseRetrieval.get_k_best_documents(number_of_articles_to_retrieve, query_embedding, self.documents)

            best_article = None

            # Get the best article from the list of retrieved articles.
            for candidate_article in best_articles:
                print(f"Candidate title: {candidate_article.title}")
                # Do not include articles in the ignore list.
                if candidate_article.title in ignored_articles: continue
                
                best_article = candidate_article
                break
            
            if best_article is None:
                raise FileNotFoundError(f"Could not find suitable article for context for use with query: {query}")
                return [], None
            
            article = best_article.title

            if verbose: print(f"Using Wikipedia article: {article} for context")

            # Retrieve all chunk embeddings from said document
            embeddings = self.get_document_embeddings(article)

        if verbose: print(f"Found {len(embeddings)} chunks for within context.")
        if verbose: print("Finding best article chunks to use as context with dense retrieval:")

        # Do not attempt to retrieve more contexts than we have
        num_contexts = min(num_contexts, len(embeddings))
        num_contexts = max(num_contexts, 0)

        best_embeddings = DenseRetrieval.get_k_best_documents(num_contexts, query_embedding, embeddings)

        if verbose: print("Context successfully retrieved.")

        retrieved_contexts = [embedding.raw_text for embedding in best_embeddings]

        # Return context as a string if only one context was retrieved
        if len(retrieved_contexts) == 1: retrieved_contexts = retrieved_contexts[0]

        return retrieved_contexts, article

        
