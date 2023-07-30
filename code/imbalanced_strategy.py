"""
imbalanced_strategy.py contains the Synthetic Minority Oversampling Technique (SMOTE) to fix imbalance issue:
    1. SMOTE 
"""
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

# SMOTE
class SMOTE():
    def __init__(self, samples, N, k):
        self.n_samples, self.n_attrs = samples.shape[0], samples.shape[1]
        self.N, self.k, self.samples, self.newindex = N, k, samples, 0

    def oversampling(self):
        N = int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1,-1), return_distance=False)[0]
            self._populate(N, i, nnarray)
        return self.synthetic

    def _populate(self, N, i, nnarray):
        for j in range(N):
            nn = random.randint(0, self.k-1)
            dif = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.random()
            self.synthetic[self.newindex] = self.samples[i] + gap*dif
            self.newindex += 1