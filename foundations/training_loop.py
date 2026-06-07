import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        #pass
        #print(x)
        #print(y)
        #print(lr)
        #print(epochs)
        #print(X.shape)
        #print(X.shape[0], X.shape[1])
        n = X.shape[0]
        #print(n)
        w = np.zeros(X.shape[1])
        #print(w)
        b = 0.0

        for _ in range(epochs):
            y_hat = X @ w + b
            #print(y_hat)
            #print(y_hat.shape,X.shape,w.shape,y.shape)
            error = y_hat - y
            dw = (2.0/n) * (X.T @ error)
            #print(f"dw = {dw}")
            db = (2.0/n) * np.sum(error)
            #print(f"db = {db}")
            w = w - lr * dw
            b = b - lr * db
        return (np.round(w,5), np.round(b,5))
        # see the difference of just X.T here and summation in Linear Regression(Training) for dw calculation. there we are calculating dw's one by one. here we calculate all dw's in one go.
