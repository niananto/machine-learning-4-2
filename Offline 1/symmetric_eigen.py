import numpy as np

# def generate_invertible_symmetric_matrix(size):
#     while True:
#         matrix = np.random.randint(low=0, high=9, size=(size, size))

#         # Check if the matrix is invertible
#         if np.linalg.matrix_rank(matrix) == size:
#             # Check if the matrix is symmetric
#             if np.allclose(matrix, matrix.T):
#                 return matrix

def generate_invertible_symmetric_matrix(size):
    # let's use the idea of strictly diagonally dominant matrix
    # they are invertible for sure
    matrix = np.random.randint(low=0, high=9, size=(size, size))
    # adding the transpose of the matrix to itself will make it symmetric
    matrix = matrix + np.transpose(matrix)
    matrix = matrix + np.diag(np.sum(np.abs(matrix), axis=1))
    return matrix

# Take the dimensions of matrix n as input.
n = input("Enter the size of the matrix: ")

# Produce a random n x n invertible symmetric matrix A. For the purpose of demonstrating, every cell of A will be an integer.
mat = generate_invertible_symmetric_matrix(int(n))
print("Matrix A: \n", mat)

# Perform Eigen Decomposition using NumPy’s library function
eigenvalues, eigenvectors = np.linalg.eig(mat)
print("Eigenvalues: \n", eigenvalues)
print("Eigenvectors: \n", eigenvectors)

# Reconstruct A from eigenvalues and eigenvectors.
A = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
print("Reconstructed A: \n", A)

# Check if the reconstruction worked properly. (np.allclose will come in handy.)
print("Is A equal to the reconstructed A? ", np.allclose(mat, A))

# Please be mindful of applying efficient methods (this will bear marks).
# You should be able to explain how your code ensures that the way you generated A ensures invertibility and symmetry.