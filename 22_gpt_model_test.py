import tiktoken
import torch

from gpt_model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size" : 50257, # 어휘 사전 크기
    "context_length": 1024, # 문맥 길이
    "emb_dim": 768, # 임베딩 차원
    "n_heads": 12, # 어텐션 헤드 개수
    "n_layers": 12, # 층 개수
    "drop_rate": 0.1, # 드롭아웃 비율
    "qkv_bias": False #쿼리, 키, 값 계산을 위한 편향
}

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)

print("입력 배치:\n", batch)
print("\n출력 크기:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 개수: {total_params:,}")

print("토큰 임베딩 총 크기:", model.tok_emb.weight.shape)
print("출력 총 크기:", model.out_head.weight.shape)

total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
print(f"가중치 묶기를 고려한 훈련 가능한 파라미터 개수: {total_params_gpt2:,}")

for i, block in enumerate(model.trf_blocks):
    ffn = block.ff
    num_params = sum(p.numel() for p in ffn.parameters())
    print(f"{i}번째 FFN의 파라미터 수:{num_params}")
    att = block.att
    att_params = sum(p.numel() for p in att.parameters())
    print(f"{i}번째 어텐션의 파라미터 수:{att_params}")

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"모델에 필요한 메모리 공간: {total_size_mb:.2f} MB")
