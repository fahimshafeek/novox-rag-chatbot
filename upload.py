import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
import uuid
import time

# Your Qdrant Cloud URL (configured in GitHub Secrets)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "novox_knowledge"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

print(f"🔌 Connecting to Qdrant server at {QDRANT_URL}...")
try:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
except Exception as e:
    print(f"❌ Failed to connect to Qdrant at {QDRANT_URL}.")
    print("Please double check that your QDRANT_URL and QDRANT_API_KEY secrets in GitHub Actions are correct.")
    raise e

# Removed FastEmbed initialization. Using Google Gemini API.
def chunk_text(text, chunk_size=150):
    """Splits large page text into smaller ~150 word chunks for better RAG retrieval."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_and_upload():
    documents = []
    metadata = []
    
    print("📖 Reading and chunking scraped data...")
    with open("scraped_data_output.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            url = record.get("url", "")
            role = record.get("role", "public")
            content = record.get("content", "")
            
            # Break the page down into small chunks
            chunks = chunk_text(content)
            
            # Attach the strict security role to EVERY single chunk
            for chunk in chunks:
                if chunk.strip():
                    documents.append(chunk)
                    metadata.append({
                        "url": url,
                        "role": role
                    })

    # CRITICAL: Prevent duplicate data by wiping the collection if it already exists
    print("🧹 Checking for existing database...")
    try:
        if client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"🗑️ Deleting old '{COLLECTION_NAME}' collection to prevent duplicate chunks...")
            client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception as e:
        print("\n❌ ERROR: Failed to communicate with Qdrant.")
        print("This usually happens if your QDRANT_URL or QDRANT_API_KEY is incorrect or outdated.")
        print("Please check your GitHub Secrets to ensure you are using the correct Qdrant Cloud URL.\n")
        raise e

    print(f"🚀 Vectorizing {len(documents)} chunks using Google Gemini Embeddings...")
    
    # Create the collection with Google's embedding size (768)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    points = []
    print(f"🚀 Vectorizing {len(documents)} chunks using Google Gemini Embeddings (embedContent)...")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
    
    for i, doc in enumerate(documents):
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": doc}]}
        }
        
        # Exponential backoff for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                emb = response.json().get("embedding", {}).get("values", [])
                
                payload_data = {"document": doc}
                payload_data.update(metadata[i])
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=emb, 
                        payload=payload_data
                    )
                )
                break
            elif response.status_code == 429:
                sleep_time = (2 ** attempt) + 2
                print(f"⚠️ Rate limit hit. Sleeping for {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                response.raise_for_status() # Raise other errors immediately (like 404)
        else:
            raise Exception(f"Failed to embed document after {max_retries} retries.")
            
        if (i + 1) % 50 == 0:
            print(f"✅ Processed {i + 1}/{len(documents)} chunks...")

    if not points:
        print("⚠️ No valid chunks to upload. Skipping database update.")
        return
        
    print(f"🚀 Uploading {len(points)} vectors to Qdrant...")
    # Upsert all points into Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print("\n✅ Upload Complete! Your data is now live on the host server.")

if __name__ == "__main__":
    process_and_upload()