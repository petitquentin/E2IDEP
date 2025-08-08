import numpy as np

def save_array_as_mtx(array, output_path):
    """
    Saves a 2D NumPy array to a file in Matrix Market (MTX) format (coordinate format).

    Parameters:
    - array (np.ndarray): 2D NumPy array to be saved.
    - output_path (str): Path to the output .mtx file.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    num_rows, num_cols = array.shape
    num_entries = num_rows * num_cols

    with open(output_path, "w") as f:
        f.write(f"{num_rows} {num_cols} {num_entries}")
        for i in range(num_rows):
            for j in range(num_cols):
                f.write(f"\n{i + 1} {j + 1} {array[i, j]}")

def read_mtx(input_path):
    """
    Reads a Matrix Market (MTX) file in coordinate format into a 2D NumPy array.

    Parameters:
    - input_path (str): Path to the .mtx file.

    Returns:
    - np.ndarray: Reconstructed 2D NumPy array.
    """
    with open(input_path, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    data_started = False
    expected_nnz = 0
    actual_nnz = 0
    matrix = None

    for line in lines:
        tokens = line.split()

        # Skip lines that don't start with a digit
        if not tokens[0].isdigit():
            continue

        # First line with dimensions and number of non-zeros
        if not data_started:
            rows, cols, expected_nnz = map(int, tokens)
            matrix = np.zeros((rows, cols))
            data_started = True
        else:
            i, j, val = int(tokens[0]), int(tokens[1]), float(tokens[2])
            matrix[i - 1, j - 1] = val  # Convert to 0-based indexing
            actual_nnz += 1

    if actual_nnz != expected_nnz:
        print(f"Warning: Mismatch in non-zero entries (expected {expected_nnz}, found {actual_nnz})")

    return matrix
