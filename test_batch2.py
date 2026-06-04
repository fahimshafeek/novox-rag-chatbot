import requests
import os
url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContentsXYZ?key=dummy"
res = requests.post(url, json={})
print("invalid status:", res.status_code)
