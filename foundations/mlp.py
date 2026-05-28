import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        #pass
        ##z1 = x @ weights[0].T + biases[0]
        #z1 = x[0] * weights[0][0].T + x[1] * weights[0][1].T + biases[0]
        #a1 = np.maximum(0,z1)
        ##z2 = a1 @ weights[1] + biases[1]
        #print(weights[1][0])
        #z2 = a1[0] * weights[1][0] + a1[1] * weights[1][1] + biases[1]
        #return np.round(z2,5)
        h = x
        #print(len(weights))
        for i in range(len(weights)):
            h = h @ weights[i] + biases[i]
            #print(weights[i],weights[i].shape,weights[i].ndim)
            #print(x @ weights[i],(x @ weights[i]).shape,(x @ weights[i]).ndim)
            if i < len(weights) - 1:
                h = np.maximum(0,h)
        #print(h)
        return np.round(h ,5)


