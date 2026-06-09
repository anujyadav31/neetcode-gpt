import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        #pass
        #print(vocabulary_size)
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16,1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        #pass
        print(f"x = {x}, x_shape = {x.shape}")
        embeddings = self.embedding_layer(x)
        print(f"embeddings = {embeddings}, embeddings_shape = {embeddings.shape}")
        averaged = torch.mean(embeddings, dim=1)
        print(f"averaged = {averaged}, averaged_shape = {averaged.shape}")
        projected = self.linear_layer(averaged)
        print(f"projected = {projected}, projected_shape = {projected.shape}")
        z = torch.round(self.sigmoid_layer(projected), decimals=4)
        print(f"output = {z}, z_shape = {z.shape}")
        return z#torch.round(self.sigmoid_layer(projected), decimals=4)
        
