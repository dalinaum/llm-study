from layer_norm import LayerNorm
import torch

ln = LayerNorm(emb_dim=5)
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("평균:\n", mean)
print("분산:\n", var)