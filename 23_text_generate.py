import tiktoken
import torch

from generate_text_simple import generate_text_simple
from gpt_model import GPTModel


tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("인코딩된 ID:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

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
model = GPTModel(GPT_CONFIG_124M)
model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("출력:", out)
print("출력 길이:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)