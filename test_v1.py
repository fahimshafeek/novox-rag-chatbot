import requests
import os
url = "https://generativelanguage.googleapis.com/v1/models/text-embedding-004:batchEmbedContents?key=dummy"
res = requests.post(url, json={"requests": [{"model": "models/text-embedding-004", "content": {"parts": [{"text": "hello"}]}}]})
print("v1 status:", res.status_code)
