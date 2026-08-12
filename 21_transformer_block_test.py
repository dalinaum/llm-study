import torch
from transformer_block import TransformerBlock

GPT_CONFIG_124M = {
    "vocab_size" : 50257, # 어휘 사전 크기
    "context_length": 1024, # 문맥 길이
    "emb_dim": 768, # 임베딩 차원
    "n_heads": 12, # 어텐션 헤드 개수
    "n_layers": 12, # 층 개수
    "drop_rate": 0.1, # 드롭아웃 비율
    "qkv_bias": False #쿼리, 키, 값 계산을 위한 편향
}

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("입력 크기:", x.shape)
print("출력 크기:", output.shape)
