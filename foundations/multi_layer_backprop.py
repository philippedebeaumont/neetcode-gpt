import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        z1 = np.dot(x, W1.T) + b1
        a1 = np.maximum(0, z1)
        y_hat = np.dot(a1, W2.T) + b2
        loss = np.round(np.mean(np.square(y_hat - y_true)), 4)

        dz2 = 2 * (y_hat - y_true)
        dw2 = np.round(dz2.reshape(1, -1) @ a1.reshape(1, -1), 4)
        db2 = np.round(dz2, 4)
        
        da1 = np.dot(dz2.reshape(1, -1), W2).flatten()
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = np.round(dz1.reshape(-1, 1) @ x.reshape(1, -1), 4)
        db1 = np.round(dz1, 4)

        return {"loss": loss, "dW1": dW1, "db1": db1, "dW2": dw2, "db2": db2}