from gpt_dataset_v1 import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

print("maxlength 4/ stride 1")
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

dataloader2 = create_dataloader_v1(
    raw_text, batch_size=1, max_length=2, stride=2, shuffle=False
)
print("maxlength 2/ stride 2")
data_iter = iter(dataloader2)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

dataloader3 = create_dataloader_v1(
    raw_text, batch_size=1, max_length=8, stride=2, shuffle=False
)
print("maxlength 8/ stride 2")
data_iter = iter(dataloader3)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("입력:\n", inputs)
print("\n타깃:\n", targets)