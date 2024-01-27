import numpy as np
import cv2
import matplotlib.pyplot as plt

# Perform Singular Value Decomposition using NumPy’s library function.
# Given a matrix A and an integer k, write a function low_rank_approximation(A, k) that returns the k-rank approximation of A.
def low_rank_approximation(A, k):
    U, D, V = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(D[:k]) @ V[:k, :]

# Take a photo of a book’s cover within your vicinity. Let’s assume it is named image.jpg. 
# Use OpenCV or similar frameworks to read image.jpg. 
# Transform it to grayscale using functions such as cv2.cvtColor(). 
# If you wish, resize to lower dimensions (~500) for faster computation. 
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# The grayscale image will be an n x m matrix A. 
print(gray.shape)

# Now vary the value of k from 1 to min(n, m) (take at least 10 such values in the interval). 
# In each case, plot the resultant k-rank approximation as a grayscale image. 
# Observe how the images vary with k. You can find a sample intended output in the shared folder. 
k_pool = [1, 5, 10, 20, 30, 40, 45, 50, 100, 200, 400, 900]
for i in range(12):
    k = k_pool[i]
    plt.subplot(3, 4, i+1)
    plt.imshow(low_rank_approximation(gray, k), cmap='gray')
    plt.title('k = ' + str(k))

plt.show()

# Find the lowest k such that you can clearly read out the author’s name from the image corresponding to the k-rank approximation.  
print("The lowest k is: 40")