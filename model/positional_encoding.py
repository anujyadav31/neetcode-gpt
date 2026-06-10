import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        #pass
        #print(f"seq_len = {seq_len}, d_model = {d_model}")
        
        PE = np.zeros((seq_len,d_model))
        #print(PE)
        position = np.arange(seq_len).reshape(-1, 1)
        #print(f"position = {position}, position_shape = {position.shape}")
        div_term = 10000 ** (np.arange(0, d_model, 2) / d_model) 
        #print(f"np.arange(0, d_model, 2) / d_model = {np.arange(0, d_model, 2) / d_model}")
        print(f"div_term = {div_term}, div_term.shape = {div_term.shape}")
        #print(f"position / div_term = {position / div_term}")
        #print(f"np.sin(position / div_term) = {np.sin(position / div_term)}")
        PE[:, 0::2] = np.sin(position / div_term)          
        PE[:, 1::2] = np.cos(position / div_term)  
        #print(div_term[:PE[:, 1::2].shape[1]])
        return np.round(PE, 5)        
        
