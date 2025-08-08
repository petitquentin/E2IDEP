
from tqdm import tqdm

def compute_covariance_matrix(data):
    """
    Computes the covariance matrix from a 2D NumPy array.
    
    Parameters:
    - data (np.ndarray): A 2D array of shape (n_samples, n_features).
    
    Returns:
    - np.ndarray: Covariance matrix of shape (n_features, n_features).
    """
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")

    n_samples, n_features = data.shape

    # Step 1: Compute the mean of each feature
    mean = np.mean(data, axis=0)

    # Step 2: Center the data by subtracting the mean
    centered_data = data - mean

    # Step 3: Compute the covariance matrix manually
    cov_matrix = np.zeros((n_features, n_features))

    # Step 4: Diagonal elements (variances)
    for i in range(n_features):
        cov_matrix[i, i] = np.sum(centered_data[:, i] ** 2) / (n_samples - 1)

    # Step 5: Off-diagonal elements (covariances)
    for i in tqdm(range(n_features), desc="Computing covariances"):
        for j in range(i + 1, n_features):
            cov = np.sum(centered_data[:, i] * centered_data[:, j]) / (n_samples - 1)
            cov_matrix[i, j] = cov_matrix[j, i] = cov

    return cov_matrix