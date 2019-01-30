import numpy as np
from findMin import findMin,findMinL1

class PCA:
    '''
    Solves the PCA problem min_Z,W (Z*W-X)^2 using SVD
    '''

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mu = np.mean(X,axis=0)
        X = X - self.mu

        U, s, Vh = np.linalg.svd(X)
        self.W = Vh[:self.k]

    def compress(self, X):
        X = X - self.mu
        Z = X@self.W.T
        return Z

    def expand(self, Z):
        X = Z@self.W + self.mu
        return X

class AlternativePCA(PCA):
    '''
    Solves the PCA problem min_Z,W (Z*W-X)^2 using gradient descent
    '''
    def fit(self, X):
        n,d = X.shape
        k = self.k
        self.mu = np.mean(X,0)
        X = X - self.mu

        # Randomly initial Z, W
        z = np.random.randn(n*k)
        w = np.random.randn(k*d)

        for i in range(10): # do 10 "outer loop" iterations
            z, f = findMin(self._fun_obj_z, z, 10, w, X, k)
            w, f = findMin(self._fun_obj_w, w, 10, z, X, k)
            print('Iteration %d, loss = %.1f' % (i, f))

        self.W = w.reshape(k,d)

    def compress(self, X):
        n,d = X.shape
        k = self.k
        X = X - self.mu
        # We didn't enforce that W was orthogonal 
        # so we need to optimize to find Z
        # (or do some matrix operations)
        z = np.zeros(n*k)
        z,f = findMin(self._fun_obj_z, z, 100, self.W.flatten(), X, k)
        Z = z.reshape(n,k)
        return Z

    def _fun_obj_z(self, z, w, X, k):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        R = np.dot(Z,W) - X
        f = np.sum(R**2)/2
        g = np.dot(R, W.transpose())
        return f, g.flatten()

    def _fun_obj_w(self, w, z, X, k):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        R = np.dot(Z,W) - X
        f = np.sum(R**2)/2
        g = np.dot(Z.transpose(), R)
        return f, g.flatten()

class RobustPCA(PCA):
    
    def fit(self, X):
        n,d = X.shape
        k = self.k
        self.mu = np.mean(X,0)
        X = X - self.mu

        # Randomly initial Z, W (for later gradient-based approach)
        z = np.random.randn(n*k)
        w = np.random.randn(k*d)
        epsilon = 0.0001
        R = np.dot(z.reshape(n, k), w.reshape(k, d)) - X
        # f = np.sum(np.sum(np.sqrt((R ** 2) + epsilon)))
        f = np.sum(np.sqrt((R ** 2) + epsilon))
        for i in range(50): # do 50 "outer loop" iterations
            f_old = f
            z, f = findMin(self._fun_obj_z, z, 10, w, X, k)
            w, f = findMin(self._fun_obj_w, w, 10, z, X, k)
            R = np.dot(z.reshape(n, k), w.reshape(k, d)) - X
            epsilon = 0.0001
            # f = np.sum(np.sum(np.sqrt((R ** 2) + epsilon)))
            f = np.sum(np.sqrt((R ** 2) + epsilon))
            print('Iteration {:2d}, loss = {}'.format(i, f))
            if f_old - f < 1e-4:
                break

        self.W = w.reshape(k, d)

    def compress(self, X):
        n,d = X.shape
        k = self.k
        X = X - self.mu
        # We didn't enforce that W was orthogonal 
        # so we need to optimize to find Z
        # (or do some matrix operations)
        z = np.zeros(n*k)
        z,f = findMin(self._fun_obj_z, z, 100, self.W.flatten(), X, k)
        Z = z.reshape(n,k)
        return Z

    
    def expand(self, Z):
        X = np.dot(Z, self.W) + self.mu
        return X

    def _fun_obj_z(self, z, w, X, k):
        n, d = X.shape
        Z = z.reshape(n, k)
        W = w.reshape(k, d)

        R = np.dot(Z, W) - X
        epsilon = 0.0001
        f = np.sum(np.sqrt((R ** 2) + epsilon))
        # f = np.sum(np.sum(np.sqrt((R**2) + epsilon)))
        dR = np.divide(R, np.sqrt((R**2) + epsilon))
        # g = dR.dot(W.T)
        g = np.dot(dR, W.transpose())
        return f, g.flatten()

    def _fun_obj_w(self, w, z, X, k):
        n, d = X.shape
        Z = z.reshape(n, k)
        W = w.reshape(k, d)

        R = np.dot(Z,W) - X
        epsilon = 0.0001
        # f = np.sum(np.sum(np.sqrt((R ** 2) + epsilon)))
        f = np.sum(np.sqrt((R ** 2) + epsilon))
        dR = np.divide(R, np.sqrt((R ** 2) + epsilon))
        g = np.dot(Z.transpose(), dR)
        return f, g.flatten()