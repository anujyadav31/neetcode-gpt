import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        #pass
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        #pass
        # Project input into Key, Query, Value spaces
        #print(f"embedded = {embedded}, embedded_shape = {embedded.shape}")
        k = self.key_gen(embedded)   # (B, T, attention_dim)
        #print(f"k = {k}, k_shape = {k.shape}")
        q = self.query_gen(embedded) # (B, T, attention_dim)
        #print(f"q = {q}, q_shape = {q.shape}")
        v = self.value_gen(embedded) # (B, T, attention_dim)
        print(f"v = {v}, v_shape = {v.shape}")

        # Attention scores: (Q @ K^T) / sqrt(d_k)
        scores = q @ torch.transpose(k, 1, 2)
        #print(f"scores = {scores}")
        context_length, attention_dim = k.shape[1], k.shape[2]
        #print(f"k.shape[1] = {k.shape[1]}, k.shape[2] = {k.shape[2]}")
        #print(f"context_length = {context_length}, attention_dim = {attention_dim}")
        scores = scores / (attention_dim ** 0.5)
        #print(f"(attention_dim ** 0.5) = {(attention_dim ** 0.5)}")
        #print(f"scores = {scores}")

        # Causal mask: prevent attending to future tokens
        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        #print(f"lower_triangular = {lower_triangular}")
        mask = lower_triangular == 0
        #print(f"lower_triangular = {lower_triangular}")
        #print(f"mask = {mask}")
        scores = scores.masked_fill(mask, float('-inf'))
        #print(f"scores = {scores}, scores_shape = {scores.shape}")
        scores = nn.functional.softmax(scores, dim=2)
        print(f"scores = {scores}, scores_shape = {scores.shape}")
        t = torch.round(scores @ v, decimals=4)
        print(f"t = {t}, t_shape = {t.shape}")
        return t
