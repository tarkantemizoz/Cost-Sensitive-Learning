import numpy as np
import cvxopt
import cvxopt.solvers

from numpy import linalg

def linear_kernel(x1, x2, input=None):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, degree=3):
    return (1 + np.dot(x, y)) ** degree

def gaussian_kernel(x, y, sigma):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel_fnc="linear", C=None, degree=None):
        self.k_fnc = kernel_fnc
        self.kernel = (linear_kernel if self.k_fnc=="linear" else (gaussian_kernel if self.k_fnc=="rbf" else polynomial_kernel))
        self.C = C
        self.degree = degree
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y, r):

        m, n = X.shape    
        for i in range(0,len(y)): 
            if y[i] == 0:
                y[i] = -1                
        y = y.reshape(-1,1) * 1.

        sigma = 1 / (n * X.var())
        self.param = (sigma if self.k_fnc == "rbf" else (self.degree if self.k_fnc == "poly" else None))

        # Gram matrix
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i,j] = self.kernel(X[i], X[j], self.param)

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(m) * -1)
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(0.0)
        costs = r[np.arange(len(r)), np.argmax(r,1)] - r[np.arange(len(r)), np.argmin(r,1)]

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(m) * -1))
            h = cvxopt.matrix(np.zeros(m))
        else:
            h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self.C * costs)))
            G = cvxopt.matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        sv_1 = a > 1e-5
        sv_2 = a <= (self.C * costs)
        sv = sv_1 == sv_2
        ind = np.arange(len(a))[sv]
        self.a_1 = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        
        #print("%d support vectors out of %d points" % (len(self.a_1), m))

        # Intercept
        self.b = 0
        for i in ind:
            self.b += y[i]
            for j in ind:
                self.b -= a[j] * y[j] * K[j,i]   
                
        self.b /= len(ind)
        self.w = np.sum(a[:, np.newaxis] * y * X, axis=0)
        self.X = X
        self.y = y
        self.a = a
        
    def project(self, X):

        y_predict = np.zeros(len(X))
        for j in range(len(X)):
            for i in range(len(self.X)):
                y_predict[j] += self.y[i] * self.a[i] * self.kernel(X[j], self.X[i], self.param)
        return y_predict + self.b

    def predict(self, X):

        preds = np.zeros(len(X))
        if self.param is None:
            for j in range(len(X)):    
                preds[j] = np.dot(X[j], self.w) + self.b         
        else:
            preds = self.project(X)
            
        preds = np.sign(preds)           
        for i in range(0,len(preds)):
            if preds[i] == -1:
               preds[i] = 0

        return preds