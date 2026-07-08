with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("총 문자 개수:", len(raw_text))
print(raw_text[:99])

import re
text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text)
result = [item for item in result if item.strip()]
print(result)

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print(len(preprocessed))

print(preprocessed[:30])