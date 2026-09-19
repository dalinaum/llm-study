from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

print("설정:", settings)
print("파라미터 딕셔너리 키:", params.keys())
