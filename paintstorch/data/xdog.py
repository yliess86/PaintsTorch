import numpy as np

from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

class xDoG(object):
    def __init__(self, gamma=0.95, phi=1e9, eps=-1e-1, k=4.5, sigma=0.3):
        self.gamma = gamma
        self.phi   = phi
        self.eps   = eps
        self.k     = k
        self.sigma = sigma

    def __call__(self, X):
        if X.shape[2] == 3:
            X  = rgb2gray(X)

        X_f1   = gaussian_filter(X, self.sigma)
        X_f2   = gaussian_filter(X, self.sigma * self.k)

        X_diff = X_f1 - self.gamma * X_f2
        X_diff = (X_diff < self.eps) * 1.0 + (X_diff >= self.eps) * \
                 (1.0 - np.tanh(self.phi * X_diff))
        X_diff = X_diff - X_diff.min()
        X_diff = X_diff / X_diff.max()
        X_diff = X_diff >= threshold_otsu(X_diff)
        X_diff = 1.0 - X_diff.astype(np.float32)

        return X_diff
