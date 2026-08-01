import torch
import torch.nn as nn

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("평균:\n", mean)
print("분산:\n", var)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("정규화된 총 출력:\n", out_norm)
print("평균:\n", mean)
print("분산:\n", var)

torch.set_printoptions(sci_mode=False)
print("평균:\n", mean)
print("분산:\n", var)