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
    BATCH_SIZE = 50
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_meta = metadata[i:i+BATCH_SIZE]
        
        requests_payload = [{"model": "models/text-embedding-004", "content": {"parts": [{"text": doc}]}} for doc in batch_docs]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={GEMINI_API_KEY}"
        
        # Add retry logic for robustness
        for attempt in range(3):
            try:
                response = requests.post(url, json={"requests": requests_payload})
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"⚠️ Rate limited by Gemini API, retrying in 10s... (Attempt {attempt+1}/3)")
                time.sleep(10)
        
        embeddings = [item["values"] for item in response.json().get("embeddings", [])]
        
        for j, emb in enumerate(embeddings):
            payload_data = {"document": batch_docs[j]}
            payload_data.update(batch_meta[j])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=emb, 
                    payload=payload_data
                )
            )
            
        time.sleep(1.5) # Gentle pause to respect Google API limits
        print(f"✅ Processed {min(i+BATCH_SIZE, len(documents))}/{len(documents)} chunks...")

    print(f"🚀 Uploading {len(points)} vectors to Qdrant...")
    # Upsert all points into Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print("\n✅ Upload Complete! Your data is now live on the host server.")

if __name__ == "__main__":
    process_and_upload()