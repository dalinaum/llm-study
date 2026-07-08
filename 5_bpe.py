from importlib.metadata import version
import tiktoken
print("tiktoken 버전:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

text2 = "Akwirw ier"

integers = tokenizer.encode(text2)
print(integers)
strings = tokenizer.decode(integers)
print(strings)