import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("어텐션 가중치:", attn_weights_2_tmp)
print("합: ", attn_weights_2_tmp.sum())

def softmax_native(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_native = softmax_native(attn_scores_2)
print("어텐션 가중치:", attn_weights_2_native)
print("합: ", attn_weights_2_native.sum())

# 동일하지만 더 안전
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("어텐션 가중치:", attn_weights_2)
print("합: ", attn_weights_2.sum())

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

# 동일하지만 더 빠름.
attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("두 번째 행의 합: ", row_2_sum)
print("모든 행의 합:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)