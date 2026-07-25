import torch
from torch import nn
from causal_attention import CausalAttention

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
            super().__init__()
            # ModuleList registers multiple submodules to the model,
            # making them trainable and savable.
            self.heads = nn.ModuleList(
                  [CausalAttention(
                        d_in, d_out, context_length, dropout, qkv_bias
                  )
                  for _ in range(num_heads)]
            )

    def forward(self, x):
        # torch.cat concatenates along an existing axis specified
        # by dim (the last axis in this case), while stack
        # creates a new axis
        return torch.cat([head(x) for head in self.heads], dim=-1)
        # cat - joins along an existing axis: two (6, 2) -> (12, 2)
        # or (6, 4). Keeps the number of dims.
        # stack - piles up along a new axis: two (6, 3) -> (2, 6, 3).
        # Adds one dim.

if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]] # step (x^6)
    )
    batch = torch.stack([inputs, inputs], dim=0)
    torch.manual_seed(123)
    context_length = batch.shape[1]
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(
         d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)