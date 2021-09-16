import numpy as np
import faiss


class FastNearestNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.k = k

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.array(X).astype(np.float32))
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        k = n_neighbors if n_neighbors else self.k
        distances, indices = self.index.search(np.array(X).astype(np.float32), k=k)
        if return_distance:
            return distances, indices
        else:
            return indices