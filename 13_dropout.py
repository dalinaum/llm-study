import torch

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

from self_attention_v2 import SelfAttention_v2

torch.manual_seed(789)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

d_in = inputs.shape[1]
d_out = 2
sa_v2 = SelfAttention_v2(d_in, d_out)

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

context_length = attn_scores.shape[0]
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

# 내적의 분산은 d_k(키 차원)에 비례해 커지므로 표준편차인
# √d_k로 나눠 분산 1을 유지한다 (스케일드 닷 프로덕트)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)

torch.manual_seed(123)
print(dropout(attn_weights))

# Dropout results may vary depending on the operating system.
