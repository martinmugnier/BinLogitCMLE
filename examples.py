import numpy as np
from CMLE import BinLogitCMLE
from matplotlib import pyplot as plt

def err_cdf(x):
    '''Logistic model : T = 3, standard normal fixed effect and one random 
    binary covariate.'''
    return 1 / (1 + np.exp(-x))

def simulate_onebinvar(n, T, beta_0):
    K = 1
    W = np.ndarray(shape=(n, T, K)) # explanatory variables
    for row in range(n):
        for period in range(T):
            W[row, period] = np.random.binomial(1, 0.5, size=K)
    Y = np.ndarray(shape=(n,3)) # outcome variable
    for i in range(n):
        fe = np.random.normal(0,1)
        Y[i,:] = np.array([float(np.random.binomial(1, err_cdf(np.dot(W[i,j], beta_0) + fe))) for j in range(3)])
    return W, Y

# Examples :
np.random.seed(12)
W, Y = simulate_onebinvar(10000, 3, np.array([1.]))
model = BinLogitCMLE(A= W, b = Y)

# Run CMLE with constant step
beta_min, beta_list = model.fit(beta_init=np.zeros(1), n_iter=100, step=0.1, epsilon = 1e-10, hessian=False, BFGS=False)
plt.plot([model.objective(elem, model.A, model.b) for elem in beta_list])
plt.show()
print("CMLE estimator (Raphson-Newton with constant step) : %s" % beta_min) # convergence is OK but slow

# Run CMLE with Hessian step
beta_min, beta_list = model.fit(beta_init=np.zeros(1), n_iter=100, step=0.1, epsilon = 1e-10, hessian=True, BFGS=False)
plt.plot([model.objective(elem, model.A, model.b) for elem in beta_list])
plt.show()
print("CMLE estimator (Raphson-Newton with hessian step) : %s" % beta_min) # convergence in one step

# Run CMLE with L-BFGS-B
beta_min, beta_list = model.fit(beta_init=np.zeros(1), n_iter=100, step=0.1, epsilon = 1e-10, hessian=False)
plt.plot([model.objective(elem, model.A, model.b) for elem in beta_list])
plt.show()
print("CMLE estimator (L-BFGS-B) : %s" % beta_min) # convergence is faster
