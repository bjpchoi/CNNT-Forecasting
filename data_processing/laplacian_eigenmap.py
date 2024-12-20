# data_processing/laplacian_eigenmap.py

import numpy as np
from scipy.spatial import distance_matrix
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

"""
Laplacian eigenmaps perform a nonlinear dimensionality reduction designed to 
preserve local structures in high-dimensional data. One common implementation
can be found via sklearn.manifold's SpectralEmbedding. The following modular
implementation enables downstream customization and flexibility if needed.

Foundational paper:
Belkin, Mikhail, and Partha Niyogi. "Laplacian eigenmaps for dimensionality reduction 
    and data representation." Neural computation 15.6 (2003): 1373-1396.

"""


class Graph:
    def __init__(self, n_neighbors=10, mode='simple', similarity_threshold=0.5):
        """
        Initializes the graph for the Laplacian eigenmap. A generalized implementation designed
        to be adaptable (for downstream models/modifications).

        Args:
            n_neighbors (int): Number of neighbors for k-NN graph.
            mode (str): 'simple' or 'complex' graph construction.
            similarity_threshold (float): Threshold for similarity in 'complex' mode.
        """
        assert mode in ['simple', 'complex'], "unsupported mode"
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        self.adj_matrix = None

    def construct_knn_graph(self, data):
        """
        Constructs a k-NN graph to approximate data topology/connectivity.

        Args:
            data (np.ndarray): Input data matrix (samples x features).

        Returns:
            np.ndarray: Adjacency matrix.
        """
        distances = distance_matrix(data, data)
        np.fill_diagonal(distances, np.inf)
        num_points = data.shape[0]
        grid_x, grid_y, _ = np.divmod(np.arange(num_points), int(np.cbrt(num_points)))
        geo_coords = np.column_stack((grid_x, grid_y))
        geo_distances = distance_matrix(geo_coords, geo_coords)
        weighted_distances = distances + geo_distances # for start; can be modified as needed
    
        knn_indices = np.argsort(weighted_distances, axis=1)[:, :self.n_neighbors]
        adj = np.zeros((data.shape[0], data.shape[0]), dtype=np.float32)
        for i, neighbors in enumerate(knn_indices):
            adj[i, neighbors] = 1
        return adj

    def construct_complex_graph(self, data):
        """
        Constructs a graph based on cosine similarity and thresholding.
        Can be modified as needed.

        Args:
            data (np.ndarray): Input data matrix (samples x features).

        Returns:
            np.ndarray: Adjacency matrix.
        """
        norm_data = data / np.linalg.norm(data, axis=1, keepdims=True)
        similarity = np.dot(norm_data, norm_data.T)
        similarity[similarity < self.similarity_threshold] = 0
        adj = (similarity > 0).astype(np.float32)
        num_points = data.shape[0]
        grid_x, grid_y, _ = np.divmod(np.arange(num_points), int(np.cbrt(num_points)))
        geo_coords = np.column_stack((grid_x, grid_y))
        geo_distances = distance_matrix(geo_coords, geo_coords)
        distances = distance_matrix(data, data)
        np.fill_diagonal(distances, np.inf)
        weighted_distances = distances + geo_distances
    
        knn_indices = np.argsort(weighted_distances, axis=1)[:, :self.n_neighbors]
        knn_adj = np.zeros_like(adj)
        for i, neighbors in enumerate(knn_indices):
            knn_adj[i, neighbors] = 1
    
        adj = np.maximum(adj, knn_adj)
        return adj

    def build_adjacency_matrix(self, data):
        """
        Builds the adjacency matrix based on the specified mode.

        Args:
            data (np.ndarray): Input data matrix (samples x features).

        Returns:
            np.ndarray: Adjacency matrix.
        """
        if self.mode == 'simple':
            adj = self.construct_knn_graph(data)
        elif self.mode == 'complex':
            adj = self.construct_complex_graph(data)
        self.adj_matrix = adj
        return adj


class LaplacianEigenmap:
    def __init__(self, n_neighbors=10, n_components=2, mode='simple', similarity_threshold=0.5):
        """
        Initializes the Laplacian eigenmap. A starter/example implementation.

        Args:
            n_neighbors (int): Number of neighbors for graph construction.
            n_components (int): Number of dimensions for embedding.
            mode (str): 'simple' or 'complex' graph construction.
            similarity_threshold (float): Threshold for similarity in 'complex' mode.
        """
        assert mode in ['simple', 'complex'], "mode not supported"
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        self.embedding_ = None

    def fit_transform(self, data):
        """
        Fits the model to the data and transforms it into the embedding space.

        Args:
            data (np.ndarray): Input data matrix (samples x features).

        Returns:
            np.ndarray: Embedded coordinates.
        """
        print("Starting Laplacian Eigenmap computation...")
        graph = Graph(n_neighbors=self.n_neighbors, mode=self.mode, similarity_threshold=self.similarity_threshold)
        adj = graph.build_adjacency_matrix(data)

        n_components, labels = connected_components(csgraph=csr_matrix(adj), directed=False, return_labels=True)
        if n_components > 1:
            print(f"Graph has {n_components} connected components. Merging components to make it connected.")
            for i in range(1, n_components):
                idx_first = np.where(labels == 0)[0][0]
                idx_other = np.where(labels == i)[0][0]
                adj[idx_first, idx_other] = 1
                adj[idx_other, idx_first] = 1
            n_components, labels = connected_components(csgraph=csr_matrix(adj), directed=False, return_labels=True)
            assert n_components == 1, "Failed to make the graph connected."

        # Degree matrix
        degrees = np.sum(adj, axis=1)
        D = np.diag(degrees)

        # Unnormalized Laplacian
        L = D - adj

        # Compute eigenvalues and eigenvectors
        print("Computing eigenvalues and eigenvectors...")
        eigenvalues, eigenvectors = eigh(L)
        print("Eigenvalues and eigenvectors computed.")

        # Select the first k non-zero eigenvectors
        # To avoid numerical issues, consider eigenvalues > 1e-5 as non-zero
        eps = 1e-5
        non_zero_indices = np.where(eigenvalues > eps)[0]
        if len(non_zero_indices) < self.n_components + 1:
            raise ValueError("Not enough non-zero eigenvalues to compute the embedding.")
        selected_indices = non_zero_indices[:self.n_components]
        self.embedding_ = eigenvectors[:, selected_indices]
        print("Laplacian Eigenmap embedding completed.")
        return self.embedding_
    
    def save_embedding(self, filepath):
        if self.embedding_ is None:
            raise ValueError("No embedding found. Please run fit_transform first.")
        np.save(filepath, self.embedding_)
        print(f"Embedding saved to {filepath}.")
