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
        #pass
        x = np.array(x) 
        W1 = np.array(W1) 
        b1 = np.array(b1) 
        W2 = np.array(W2) 
        b2 = np.array(b2) 
        y_true = np.array(y_true)

        print(f"x = {x}, shape = {x.shape}, dim = {x.ndim}") 
        print(f"W1 = {W1}, shape = {W1.shape}, dim = {W1.ndim}") 
        print(f"b1 = {b1}, shape = {b1.shape}, dim = {b1.ndim}") 
        print(f"W2 = {W2}, shape = {W2.shape}, dim = {W2.ndim}") 
        print(f"b2 = {b2}, shape = {b2.shape}, dim = {b2.ndim}") 
        print(f"y_true = {y_true}, shape = {y_true.shape}, dim = {y_true.ndim}") 

        z1 = x @ W1.T + b1
        print(f"z1 = {z1}, shape = {z1.shape}, dim = {z1.ndim}")
        a1 = np.maximum(0,z1)
        print(f"a1 = {a1}, shape = {a1.shape}, dim = {a1.ndim}")
        print(f"W2 = {W2}, shape = {W2.shape}, dim = {W2.ndim}")
        z2 = a1 @ W2.T + b2
        print(f"z2 = {z2}, shape = {z2.shape}, dim = {z2.ndim}")
        loss = np.mean(np.square(z2 - y_true))
        print(f"loss = {loss}, shape = {loss.shape}, dim = {loss.ndim}")

        print(f"y_true = {y_true},len = {len(y_true)} , shape = {y_true.shape}, dim = {y_true.ndim}")
        n = len(y_true) if y_true.ndim > 0 else 1
        dz2 = (2 / n )* (z2 - y_true) # dL/dz2
        print(f"dz2 = {dz2}, shape = {dz2.shape}, dim = {dz2.ndim}")
        dW2 = dz2.reshape(-1,1) @ a1.reshape(1,-1) # dl/dW2 = dl/dz2 x dz2/dW2
        print(f"dW2 = {dW2}, shape = {dW2.shape}, dim = {dW2.ndim}")
        db2 = dz2
        print(f"db2 = {db2}, shape = {db2.shape}, dim = {db2.ndim}") 
        da1 = dz2 @ W2
        print(f"da1 = {da1}, shape = {da1.shape}, dim = {da1.ndim}")
        dz1 = da1 * (z1 > 0).astype(float)
        print(f"dz1 = {dz1}, shape = {dz1.shape}, dim = {dz1.ndim}")
        dW1 = dz1.reshape(-1,1) @ x.reshape(1,-1)
        print(f"dW1 = {dW1}, shape = {dW1.shape}, dim = {dW1.ndim}")
        db1 = dz1
        print(f"db1 = {db1}, shape = {db1.shape}, dim = {db1.ndim}")

        #return {
        #    'loss':  np.round(loss, 4),
        #    'dW1':   np.round(dW1, 4),
        #    'db1':   np.round(db1, 4),
        #    'dW2':   np.round(dW2, 4),
        #    'db2':   np.round(db2, 4)
        #}
        return {
            'loss': np.round(loss, 4),#round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }

