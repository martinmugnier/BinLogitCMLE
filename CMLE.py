# -*- coding: utf-8 -*-
"""
%------------------- CMLE for Binary fixed effects models ---------------------
 Papers : 
     - 'On the Existence of Conditional Maximum Likelihood Estimates of 
         the Binary Logit Modelwith Fixed Effects' ; 
     - 'Fixed Effects Binary Choice Models with Three or More Periods' (with 
                                            X. D'Haultfoeuille & L. Davezies).
 Author : Martin MUGNIER
 Version : 09/20/2020
 This code implements the Conditional Maximum Likelihood Estimator (C.M.L.E.) 
 of the (normalized) slope parameter in a panel binary logistic model with 
 individual fixed effects. It includes an automated test of CMLE's existence 
 condition before starting the optimization for small samples (n<100). To test
 the condition in larger samples, please use the 'separation_test()' command.
%----------------------------------------------------------------------------

 References :
----------
 Woolridge, "Econometric Analysis of Cross-Section and Panel Data" (Chap 15.8.3).
 
 Examples :
----------
np.random.seed(12)
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
"""

import cvxpy as cp
import numpy as np
from scipy import stats
from sympy.utilities.iterables import multiset_permutations
from scipy.special import comb
from scipy.optimize import fmin_l_bfgs_b

class BinLogitCMLE():
    """
    Implement the conditional maximum likelihood estimator of the binary logit
    model with fixed effects (T balanced periods).
    
    Reference : Woolridge, "Econometric Analysis of Cross-Section and Panel Data" (Chap 15.8.3).
    
    Inputs
    ------------
    `A': (n x T x K) numpy.array that contains individuals' covariates at each time period;
    `b': (n x T) numpy.array that contains individuals' choices at each time period.
    """
    
    def __init__(self, A, b):
        self.A = A
        # make sure b is of correct format
        # make sure it contains only ones
        # make sure A is numeric
        self.b = b
        if len(A.shape) != 3:
            print('The design matrix A must have shape (N x T x K).')
        else:
            self.n, self.T, self.K = A.shape
        
        self.R = self.compute_perm()
        if self.n<= 100:
            existence_cond = self.separation_test(verbose=False)
            if existence_cond==False:
                print("Warning ! Data violate CMLE's existence conditions.")
                            
    def compute_perm(self):
        """
        Returns
        `R' : numpy.ndarray that contains all vectors of 1/0 of size T such 
        that their coordinates sum equals k, for k in [0,T]
        """
        R = list()
        for k in range(self.T + 1):
            array = np.ndarray(shape=(int(comb(self.T, k)), self.T))
            select = np.zeros(self.T)
            select[:k] = np.ones(k)
            for idx, p in enumerate(multiset_permutations(select)):
                array[idx] = p
            R.append(array)
        return R

    def objective_i(self, i, A, b, beta):
        """
        Returns the i-th contribution to the objective function 
        (CMLE is a finite-sum).
        
        Inputs
        ------------
        `A' : design matrix;
        `b' : outcome variable;
        """
        R = self.R
        Xprime_beta = A[i].dot(beta)
        n_i = int(np.sum(b[i]))
        omega_i = np.sum(np.exp(R[n_i].dot(Xprime_beta)), axis=0)
        res = np.dot(b[i], Xprime_beta) - np.log(omega_i)
        return res
    
    def objective(self, beta, A, b):
        objective = 0
        for i in range(self.n):
            objective += self.objective_i(i, A, b, beta)
        return objective / self.n
    
    def loss(self, beta, A, b):
        objective = 0
        for i in range(self.n):
            objective += self.objective_i(i, A, b, beta)
        return - objective / self.n
    
    def loss_grad(self, beta, A, b):
        """
        Computes the global gradient.
        """
        g = np.zeros_like(beta)
        for i in range(self.n):
            g += self.comp_grad_i(i, beta)
        return - g / self.n
    
    def comp_grad_i(self, i, beta): 
        """
        Returns the pointwise gradient evaluated on datum i.
        """
        A = self.A
        b = self.b
        R = self.R
        n_i = int(np.sum(b[i]))
        Xprime_beta = A[i].dot(beta)
        if ((n_i!=0) & (n_i!=self.T)):
            omega_i = (1. / (np.sum(np.exp(R[n_i].dot(Xprime_beta)))))
            g = (np.dot(b[i], A[i]) - np.sum(np.dot(R[n_i], A[i]) * 
                        np.tile(np.exp(R[n_i].dot(Xprime_beta)), 
                                (self.K, 1)).T, axis=0) * omega_i)
        else: 
            g = np.zeros(self.K)
        return g
       
    def comp_hessian_i(self, i, beta):
        """
        Returns the pointwise Hessian matrix evaluated on datum i.
        """
        A = self.A
        b = self.b
        R = self.R
        n_i = int(np.sum(b[i]))
        Xprime_beta = A[i].dot(beta)
        if ((n_i!=0) & (n_i!=self.T)):
            omega_i = (1. / (np.sum(np.exp(R[n_i].dot(Xprime_beta)))))
            b_i = np.sum(np.dot(R[n_i], A[i]) * 
                        np.tile(np.exp(R[n_i].dot(Xprime_beta)), (self.K, 1)).T, axis=0)
            hess = (np.outer(b_i, b_i) * omega_i ** 2 - 
                    np.sum([np.outer(elem, elem) * np.exp(R[n_i].dot(Xprime_beta))[idx] for idx, elem in enumerate(np.dot(R[n_i], A[i]))]) 
                    * omega_i)
        else:
           hess= np.zeros((self.K, self.K))
        return hess

    def comp_gradient(self, beta, A, b):
        """
        Computes the global gradient.
        """
        g = np.zeros_like(beta)
        for i in range(self.n):
            g += self.comp_grad_i(i, beta)
        return g / self.n

    def comp_hessian(self, beta): # warning : one-dimensional case
        """Computes the global Hessian.
        """
        hess = np.ndarray(shape=(self.K,self.K), buffer=np.zeros(self.K**2))
        for i in range(self.n):
            hess = hess + self.comp_hessian_i(i, beta)
        return hess / self.n

    def fit(self, beta_init, n_iter=1000, step=0.01, tol=1e-6, epsilon=1e-10, hessian=False, verbose=True, BFGS=True):
        """Fit the conditional logit using either BFGS algorithm or 
        the Newton-Raphson algorithm (with or without an Hessian step).
        """
        beta = beta_init.copy()
        beta_list = [beta_init]
        if BFGS:
            beta, f_min, _ = fmin_l_bfgs_b(self.loss, beta_init, self.loss_grad, args=(self.A, self.b), pgtol=1e-6, factr=1e-30)
        elif hessian:
            while True:
                beta_t = beta - np.linalg.inv(self.comp_hessian(beta)).dot(self.comp_gradient(beta, self.A, self.b))
                t = abs(beta_t - beta)
                if t < tol:
                    break
                beta = beta_t
                #beta_list.append(beta)
        else:
            stop = True
            for t in range(n_iter):
                if stop:
                    beta = beta + step * self.comp_gradient(beta, self.A, self.b)
                    if verbose:
                        print("Iteration %s completed" % t)
                    beta_list.append(beta)
                    if self.objective(beta, self.A, self.b, ) -self.objective( beta_list[t], self.A, self.b,) < epsilon:
                        stop = False
        return beta, beta_list
    
    def AsympVariance(self, beta, score=True):
        avar = np.ndarray(shape=(self.K,self.K), buffer=np.zeros(self.K**2))
        if score:
            for i in range(self.n):
                grad = self.comp_grad_i(i, beta)
                avar += np.outer(grad, grad)
        else:
            for i in range(self.n):
                avar += - self.comp_hessian_i(i, beta) # Fix the minus ? -> get negative variance
        return np.linalg.inv(avar / self.n)
    
    def lr_nulltest(self, bet_hat, lvl=0.05):
         """
         Perform the LR test of the global null.
         ----------
         Outputs : LR statistics, p-value
         """
         lu = self.objective(bet_hat, self.A, self.b)
         lr = self.objective(np.zeros(self.K), self.A, self.b)
         ratio = 2 *self.n * (lu - lr)
         pval = stats.chi2.sf(ratio, self.K)
         if pval<lvl:
             print("H0 : \beta_j =0 \forall j is rejected at level : ", lvl)
         return ratio, pval
     
    def separation_test(self, tol=1e-4, verbose=True):
        """
        Verify that CMLE's existence condition is met (False=CMLE does not exist).
        """
        res = True
        R = self.R
        rn = np.sum([len(R[int(np.sum(self.b[i]))]) for i in range(self.n)])
        u = cp.Variable(rn)
        a = []
        for i in range(self.n):
            for j in range(len(R[int(np.sum(self.b[i]))])):
                vec = R[int(np.sum(self.b[i]))][j] - self.b[i]
                a.append(np.sum(vec[:,None] * self.A[i]))
        a = np.array(a)
        prob = cp.Problem(cp.Minimize(cp.norm(cp.sum(cp.multiply(u,a), 
                                                axis=0))), [u>=np.ones(rn)])
        prob.solve()
        if np.abs(prob.value)>tol:
            res = False
        if verbose:
            print('CMLE does exist: {}.'.format(res))
        return res