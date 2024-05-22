import time

import torch


def differential_pca(X, k):
    """
    Perform PCA on the dataset X and reduce it to k dimensions.

    Args:
    - X (torch.Tensor): The input data tensor of shape (n_samples, n_features).
    - k (int): The number of principal components to keep.

    Returns:
    - X_reduced (torch.Tensor): The data projected onto the top k principal components.
    - components (torch.Tensor): The top k principal components (eigenvectors).
    - explained_variance (torch.Tensor): The eigenvalues corresponding to the top k principal components.
    """
    # Step 1: Center the data
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean

    # Step 2: Compute the covariance matrix
    cov_matrix = torch.mm(X_centered.t(), X_centered) / (X_centered.size(0) - 1)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix, UPLO='U')

    # Step 4: Sort eigenvalues and eigenvectors
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Select the top k eigenvectors (principal components)
    components = eigenvectors[:, :k]
    explained_variance = eigenvalues[:k]

    # Step 6: Project the data onto the top k principal components
    X_reduced = torch.mm(X_centered, components)

    return X_reduced, components, explained_variance


# Example usage
if __name__ == '__main__':
    # Generate some random data
    torch.manual_seed(42)
    X = torch.randn(12800, 128).requires_grad_()

    # Perform PCA to reduce to 2 dimensions
    start = time.perf_counter()
    X_reduced, components, explained_variance = differential_pca(X, k=10)
    end = time.perf_counter()

    print("Reduced Data Shape:", X_reduced.shape)
    print("Principal Components Shape:", components.shape)
    print("Explained Variance:", explained_variance)
    print(end - start)
