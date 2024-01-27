import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import multivariate_normal
import sys

def PCA_with_SVD(data, k=2):
    if data.shape[1] < k:
        print('k must be less than feature dimension; returning original data')
        return data
    
    # mean center the data
    data -= np.mean(data, axis=0)
    
    # calculate the covariance matrix
    cov = np.cov(data, rowvar=False)
    
    # perform singular value decomposition
    U, S, V = np.linalg.svd(cov)
    
    # take the first k columns of U
    U_reduced = U[:,:k]
    
    # transform the data into the reduced subspace
    return np.dot(U_reduced.T, data.T).T

# k is the number of clusters/components
def GMM_est_with_EM(data, alias, k=4, max_iter=1000, trials=5):
    n, m = data.shape
    epsilon = 1e-6
    
    best_log_likelihood = float('-inf')
    best_phi = None
    best_mu = None
    best_sigma = None
    best_log_likelihood_history = None
    
    for trial in range(1, trials+1):
        print(f'\tTrial {trial}')
        
        def update(i):
            plt.clf()  # clear the current figure
            assignments = np.argmax(gamma_history[i], axis=1)
            for j in range(k):
                plt.scatter(data[assignments == j, 0], data[assignments == j, 1])
                plt.scatter(mu_history[i][:, 0], mu_history[i][:, 1], marker='x', c='black')
                
                # Draw contour lines for the Gaussian distribution
                x, y = np.meshgrid(np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100),
                                np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100))
                pos = np.dstack((x, y))
                rv = multivariate_normal(mu_history[i][j], sigma_history[i][j])
                plt.contour(x, y, rv.pdf(pos))
                
            plt.title(f'Iteration {i+1}')
        
        # randomly initialize the parameters
        phi = np.ones(k) / k # mixing coefficients / weights
        mu = np.random.rand(k, m)
        sigma = np.array([np.eye(m)] * k)
        log_likelihood = 0
        
        # store the histories
        mu_history = []
        sigma_history = []
        gamma_history = []
        log_likelihood_history = []
        
        # iterate until convergence
        for iter in range(max_iter):
            # E-step
            # gamma is the responsibility matrix / posterior probabilities
            gamma = np.zeros((n, k))
            for j in range(k):
                gamma[:,j] = phi[j] * multivariate_normal.pdf(data, mean=mu[j], cov=sigma[j])
            gamma /= np.sum(gamma, axis=1).reshape(-1,1)
            
            # M-step
            N = np.sum(gamma, axis=0)
            phi = N / n
            mu = np.dot(gamma.T, data) / N.reshape(-1,1)
            for j in range(k):
                sigma[j] = np.dot((data - mu[j]).T, np.dot(np.diag(gamma[:,j]), (data - mu[j]))) / N[j] + epsilon * np.eye(m)
        
            # calculate the log-likelihood
            log_likelihood_new = 0
            for j in range(k):
                log_likelihood_new += np.sum(gamma[:,j] * np.log(phi[j] * multivariate_normal.pdf(data, mean=mu[j], cov=sigma[j]) + epsilon))
            
            # store the log-likelihood
            mu_history.append(mu.copy())
            sigma_history.append(sigma.copy())
            gamma_history.append(gamma.copy())
            log_likelihood_history.append(log_likelihood_new)
            
            # check for convergence
            if np.abs(log_likelihood_new - log_likelihood) < epsilon:
                break
            log_likelihood = log_likelihood_new
            
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_phi = phi
                best_mu = mu
                best_sigma = sigma
                best_log_likelihood_history = log_likelihood_history
                
        # save the animation
        fig = plt.figure()
        ani = FuncAnimation(fig, update, frames=iter, repeat=True)
        ani.save(f'plots/animations/{alias}_k{k}_trial{trial}.gif', writer=PillowWriter(fps=2))
        plt.close()
                    
    # return the parameters
    return best_log_likelihood, best_phi, best_mu, best_sigma, best_log_likelihood_history, gamma

# taking dataset, seed, min k, max k, max iterations, number of trials as command line input
dataset, alias = sys.argv[1:3]
seed, min_k, max_k, max_iter, trials = map(int, sys.argv[3:])

# seed the random number generator
np.random.seed(seed)

# load the data
data = np.loadtxt(dataset, delimiter=',', skiprows=0)
print(data.shape)

compressed_data = PCA_with_SVD(data, k=2)
print(compressed_data.shape)

# plot the compressed data
plt.scatter(compressed_data[:,0], compressed_data[:,1])
plt.savefig(f'plots/{alias}.png')
plt.close()

# run the EM algorithm for multiple values of k
results = {}
best_k = None
best_log_likelihood = float('-inf')
for k in range(min_k, max_k+1):
    print(f'Running EM algorithm for k={k}')
    
    log_likelihood, phi, mu, sigma, log_likelihoods, gamma = GMM_est_with_EM(compressed_data, alias, k, max_iter, trials)
    results[k] = (log_likelihood, phi, mu, sigma, log_likelihoods)
    
    if log_likelihood > best_log_likelihood:
        best_log_likelihood = log_likelihood
        best_k = k
    
    # Plot the results with contour plots
    plt.figure()
        
    # Assign each point to the cluster with the highest posterior probability
    assignments = np.argmax(gamma, axis=1)
    
    
    # Get the color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for j in range(k):
        color = color_cycle[j % len(color_cycle)]  # Use modulo to prevent index out of range
        plt.scatter(compressed_data[assignments == j, 0], compressed_data[assignments == j, 1], color=color)
        x, y = np.meshgrid(np.linspace(np.min(compressed_data[:, 0]), np.max(compressed_data[:, 0]), 100),
                           np.linspace(np.min(compressed_data[:, 1]), np.max(compressed_data[:, 1]), 100))
        pos = np.dstack((x, y))
        rv = multivariate_normal(mu[j], sigma[j])
        plt.contour(x, y, rv.pdf(pos), colors=color)
    plt.scatter(mu[:, 0], mu[:, 1], marker='x', c='black')
    plt.savefig(f'plots/{alias}_k{k}_contour.png')
    plt.close()
    
# plot the log-likelihoods
plt.figure()
plt.plot(list(results.keys()), [v[0] for v in results.values()], marker='x')
plt.savefig(f'plots/{alias}_log_likelihoods.png')   
plt.close() 

# plot the log-likelihoods for the best k
if best_k:
    plt.figure()
    plt.plot(results[best_k][4], marker='x')
    plt.savefig(f'plots/{alias}_k{best_k}_log_likelihoods.png')
    plt.close()