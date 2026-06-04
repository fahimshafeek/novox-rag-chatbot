import requests
import os
url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key=invalid_key"
res = requests.post(url, json={"requests": [{"model": "models/text-embedding-004", "content": {"parts": [{"text": "hello"}]}}]})
print(res.status_code, res.text)
