import numpy as np
from sklearn.cluster import SpectralClustering

# Example graph adjacency matrix (unweighted)
adjacency_matrix = np.array([[0, 1, 1, 0],
                             [1, 0, 1, 0],
                             [1, 1, 0, 1],
                             [0, 0, 1, 0]])

# Perform spectral clustering to partition the graph into 2 parts
model = SpectralClustering(n_clusters=2, affinity='precomputed')
labels = model.fit_predict(adjacency_matrix)

print("Spectral clustering labels:", labels)
