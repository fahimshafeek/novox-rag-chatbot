import requests
import os
url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key=" + os.getenv("GEMINI_API_KEY", "dummy")
res = requests.post(url, json={"requests": [{"model": "models/text-embedding-004", "content": {"parts": [{"text": "hello"}]}}]})
print("batchEmbedContents status:", res.status_code)

url2 = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedText?key=" + os.getenv("GEMINI_API_KEY", "dummy")
res2 = requests.post(url2, json={"requests": [{"model": "models/text-embedding-004", "content": {"parts": [{"text": "hello"}]}}]})
print("batchEmbedText status:", res2.status_code)
