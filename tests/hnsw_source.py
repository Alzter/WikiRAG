import numpy as np
import heapq
from collections import deque
import random
import os
import math
from collections import defaultdict

def _euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

class HNSW():
    def __init__(self, ef_Construction, mL, M, Mmax = int, corpus = list):
        # self.hnsw = hnsw        # Multilayer graph
        self.M = M              # Number of established connections
        self.Mmax = Mmax
        self.corpus = corpus    # Embedding vector database
        self.layers = defaultdict(list)
        self.ef_Construction = ef_Construction
        self.mL = mL
        self.entry_point = None

    def build_hnsw(self):
        """
        Insert all vectors from self.corpus into the HNSW graph.
        """
        for idx, vec in enumerate(self.corpus):
            self.insert_element(vec)  # Insert each vector from the corpus


    def insert_element(self, q):
        """
        Algorithm 1:
            - Network Construction: organized via consecutive insertions of the stored elements into the graph structure.
            - For every inserted element an integer maximum layer l is randomly selected with an exponentially decaying probability distribution 
            - This algorithm adds a new point q into the HNSW graph. 
            It uses a multi-layer graph structure where higher levels contain fewer points, allowing for faster traversal and insertion.
            - The point is inserted into the top level first, and then bidirectional connections are established between the point and its neighbors on each layer.
        """
        W = []  # List for the currently found nearest elements
        ep = self.entry_point  # Get the entry point
        L = self._get_level(ep) if ep else 0  # Top layer for HNSW
        l = self._generate_level(self.mL)  # New element's level

        # for l_c in range(L,l+1):
        for l_c in range(L, l - 1, -1):  # Fixed range
            W = self.search_layer(q, ep, ef = 1, l_c = l_c)
            ep = self._get_nearest(q, W)     # Get the nearest element from W to q

        # for l_c in range(min(L,l), 0, -1):
        for l_c in range(min(L, l), -1, -1):
            W = self.search_layer(q, ep, self.ef_Construction, l_c)
            neighbors = self.select_neighbors_simple(q, self.M, l_c)
            # neighbors = self.select_neighbors_heuristic(query, )
            self._add_bidirectional_connections(q, neighbors, l_c)
        for e in neighbors:
                eConn = self._get_neighborhood(e, l_c)
                if len(eConn) > self.Mmax:
                    eNewConn = self.select_neighbors_simple(e, eConn, self.Mmax, l_c)
                    self._set_neighborhood(e, l_c, eNewConn)

        if l > L:
            self.entry_point = q    # Update hnsw inserting element q
            
    def search_layer(self, q, ep, ef, l_c):
        """
        Algorithm 2:
        - This algorithm performs a greedy search for the nearest neighbors in a given layer. 
        It starts with an entry point ep and expands to find the nearest neighbors to the query point q.
        """
        v = {ep}  # Set of visited elements
        C = [ep]  # Set of candidates
        W = [ep]  # Dynamic list of found nearest neighbors

        while len(C) > 0:
            c = C.pop()     # Extract nearest element from C to q

            f = max(W, key=lambda x: self._distance(q, x))      # Get furthest element from W to q
            if self._distance(c, q) > self._distance(f, q):
                break   # All elements in W are evaluated
            
            for e in self._get_neighborhood(c, l_c):    # Update C and W
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

    def select_neighbors_heuristic(self, query, C, M, l_c, extend_candidate, keep_prunned_connections):
        """
        Algorithm 4:
        - This algorithm extends the candidate list by including neighbors of candidates and applies a heuristic to select the M best neighbors. 
        Optionally, it can also keep some pruned connections.
        """
        raise NotImplemented
    
    def k_nn_search(self, q, k, ef):
        """
        Algorithm 5:
        - This is the search algorithm that performs the k-nearest neighbors search 
        by traversing the HNSW graph from the top layer down to the bottom, refining the search at each level.
        """
        # Algorithm 5 - K-NN-SEARCH
        W = []  # Set of current nearest elements
        ep = self.entry_point
        L = self._get_level(ep) if ep else 0

        for l_c in range(L, 1, -1):
            W = self.search_layer(q, ep, ef=1, l_c = l_c)
            ep = self._get_nearest(q, W)

        W = self.search_layer(q, ep, ef, l_c=0)
        return self.select_neighbors_simple(q, W, k)#, level=0)


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

    


# def _get_nearest(q,W):
#     """ Get nearest distance from element in the list W to the query 'q' """
#     return min(W, key=lambda x: _euclidean_distance(q, x))

