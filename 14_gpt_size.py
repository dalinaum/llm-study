from multi_head_attention import MultiHeadAttention

d_in, d_out = 768, 768
context_length = 1024
num_heads = 12
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)
