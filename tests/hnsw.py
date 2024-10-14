import numpy as np
import heapq
from collections import deque
import random
import os
import math
from collections import defaultdict

# import sys;sys.path.append("./rag")
# from retrieval import Retrieval

def _euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

class HNSW():
    def __init__(self, ef_Construction, mL, M, Mmax=int, corpus=list):
        self.M = M              # Number of established connections
        self.Mmax = Mmax        # Maximum number of connections
        self.corpus = corpus    # Embedding vector database
        self.layers = defaultdict(list)  # Multilayer graph structure
        self.ef_Construction = ef_Construction
        self.mL = mL
        self.entry_point = None

    def build_hnsw(self):
        """
        Insert all vectors from self.corpus into the HNSW graph.
        """
        for idx, vec in enumerate(self.corpus):
            # print(vec)
            self.insert_element(vec)  # Insert each vector from the corpus
    
    def insert_element(self, q):
        """
        Algorithm 1 - Insert the vector 'q' into the HNSW graph.
        """
        W = []  # List for the currently found nearest elements
        ep = self.entry_point  # Get the entry point
        L = self._get_level(ep) if ep else 0  # Top layer for HNSW
        l = self._generate_level(self.mL)  # New element's level

        for l_c in range(L, l - 1, -1):  # Fixed range
            W = self.search_layer(q, ep, ef=1, l_c=l_c)
            ep = self._get_nearest(q, W)

        for l_c in range(min(L, l), -1, -1):
            W = self.search_layer(q, ep, self.ef_Construction, l_c)
            neighbors = self.select_neighbors_simple(q, W, self.M)  # Fixed parameters
            self._add_bidirectional_connections(q, neighbors, l_c)
            for e in neighbors:
                eConn = self._get_neighborhood(e, l_c)
                if len(eConn) > self.Mmax:
                    eNewConn = self.select_neighbors_simple(e, eConn, self.Mmax)
                    self._set_neighborhood(e, l_c, eNewConn)

        if l > L:
            self.entry_point = q  # Update entry point

    def search_layer(self, q, ep, ef, l_c):
        """
        Algorithm 2: SEARCH-LAYER - Perform search on a specific layer.
        """
        v = {ep}  # Set of visited elements
        C = [ep]  # Set of candidates
        W = [ep]  # Dynamic list of found nearest neighbors

        while len(C) > 0:
            c = C.pop()  # Extract nearest element from C to q
            f = max(W, key=lambda x: self._distance(q, x))  # Get furthest element from W to q
            if self._distance(c, q) > self._distance(f, q):
                break  # All elements in W are evaluated

            for e in self._get_neighborhood(c, l_c):  # Update C and W
                if e not in v:
                    v.add(e)
                    f = max(W, key=lambda x: self._distance(q, x))
                    if self._distance(e, q) < self._distance(f, q) or len(W) < ef:
                        C.append(e)
                        W.append(e)
                        if len(W) > ef:
                            f = max(W, key=lambda x: self._distance(q, x))
                            W.remove(f)
        return W
    
    def select_neighbors_simple(self, q, C, M):
        """
        Algorithm 3:
        - This is a simple selection mechanism that returns the M closest neighbors to the query point q from a candidate list C.
        """
        C = sorted(C, key=lambda x: self._distance(q, x))
        return C[:M]
    
    def k_nn_search(self, query_vec, k, ef):
        """

        Algorithm 5: K-NN-SEARCH - Search for k nearest neighbors to 'query_vec'.
        """
        W = []  # Set of current nearest elements
        ep = self.entry_point
        L = self._get_level(ep) if ep else 0

        for l_c in range(L, 1, -1):
            W = self.search_layer(query_vec, ep, ef=1, l_c=l_c)
            ep = self._get_nearest(query_vec, W)

        W = self.search_layer(query_vec, ep, ef, l_c=0)
        return self.select_neighbors_simple(query_vec, W, k)

    """
    Helper methods like _add_bidirectional_connections, _distance, etc. remain the same
    """
    """
    Helper function
    """
    def _add_bidirectional_connections(self, q, neighbors, level):
            for neighbor in neighbors:
                self._add_connection(q, neighbor, level)
                self._add_connection(neighbor, q, level)

    def _add_connection(self, point1, point2, level):
        if point2 not in self.layers[level]:
            self.layers[level].append(point2)

    def _generate_level(self,mL):
        """ For every inserted element an integer maximum layer l is randomly selected with an exponentially decaying probability distribution
        normalized by the mL parameter
        """  
        return int(math.floor(-math.log(random.uniform(0, 1)) * mL))

    def _get_level(self, point):
        return len(self.layers) - 1 if point in self.layers else 0
    
    def _get_neighborhood(self, e, level):
        return [neighbor for neighbor in self.layers[level] if neighbor != e]

    def _set_neighborhood(self, e, level, neighbors):
        self.layers[level] = neighbors

    def _get_nearest(self, q, W):
        """ Get nearest distance from element in the list W to the query 'q' """
        return min(W, key=lambda x: self._distance(q, x))
    
    def _distance(self, vec1, vec2):
        """ Using Euclidean Distance to calculate distance"""
        return _euclidean_distance(vec1, vec2)
        # return np.linalg.norm(np.array(vec1) - np.array(vec2))

# Usage Example:
# rag_path = os.path.abspath(os.path.join('.', 'context\knowledge_base'))
# print(rag_path)  # Check if this is the correct folder
# .\context\knowledge_base\A Clockwork Orange (novel)\chunk_10.txt

# retrieval = Retrieval(".\context\knowledge_base")
# print(len(retrieval.documents))
# print(retrieval)



# List of vector embeddings
# corpus = np.random.rand(100, 128)  # 100 vectors, each of size 128 dimensions

# Initialize HNSW
# hnsw = HNSW(ef_Construction=200, mL=1.5, M=5, Mmax=10, corpus=corpus)

# Build the HNSW index from the corpus
# hnsw.build_hnsw()

# # Perform a search for k-nearest neighbors
# query_vector = np.random.rand(128)  # Query vector of size 128 dimensions
# k_neighbors = hnsw.k_nn_search(query_vector, k=5, ef=50)

# print("Nearest Neighbors:", k_neighbors)

