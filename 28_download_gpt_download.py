import urllib.request


url = (
    "https://raw.githubusercontent.com/rickiepark/"
    "llm-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)