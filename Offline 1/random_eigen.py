import numpy as np

# def generate_invertible_matrix(size):
#     while True:
#         matrix = np.random.randint(low=0, high=9, size=(size, size))

#         # Check if the matrix is invertible
#         if np.linalg.matrix_rank(matrix) == size:
#             return matrix
    
def generate_invertible_matrix(size):
    # generate a strictly diagonally dominant matrix
    # https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
    # they are invertible for sure
    matrix = np.random.randint(low=0, high=9, size=(size, size))
    matrix = matrix + np.diag(np.sum(np.abs(matrix), axis=1))
    return matrix
    
# Take the dimensions of matrix n as input.
# Produce a random n x n invertible matrix A. For the purpose of demonstrating, every cell of A will be an integer.
n = input("Enter the size of the matrix: ")
mat = generate_invertible_matrix(int(n))
print("Matrix A: \n", mat)

# Perform Eigen Decomposition using NumPy’s library function 
eigenvalues, eigenvectors = np.linalg.eig(mat)
print("Eigenvalues: \n", eigenvalues)
print("Eigenvectors: \n", eigenvectors)

# Reconstruct A from eigenvalues and eigenvectors
# A = V * Diag(Lambda) * V^-1
A = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
A = np.real(np.round(A, 0)) # as we are taking integers strictly, this won't be a problem
print("Reconstructed A: \n", A)

# Check if the reconstruction worked properly. (np.allclose will come in handy.)
print("Is A equal to the reconstructed A? ", np.allclose(mat, A))