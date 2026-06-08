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
except Exception:
    print(f"❌ Failed to connect to Qdrant at {QDRANT_URL}.")
    print("Please double check that your QDRANT_URL and QDRANT_API_KEY secrets in GitHub Actions are correct.")
    raise Exception("Qdrant connection failed.")

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

    print("🧹 Checking for existing database...")
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"🆕 Creating new '{COLLECTION_NAME}' collection...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
        else:
            print(f"🔄 Database exists. Preparing for incremental update...")
    except Exception:
        print("\n❌ ERROR: Failed to communicate with Qdrant.")
        print("This usually happens if your QDRANT_URL or QDRANT_API_KEY is incorrect or outdated.")
        print("Please check your GitHub Secrets to ensure you are using the correct Qdrant Cloud URL.\n")
        raise Exception("Failed to prepare database.")

    # SMART INCREMENTAL UPDATE: Delete old data ONLY for pages we successfully scraped
    # This ensures if the crawler crashes halfway, we don't lose the rest of the database!
    
    # Qdrant requires a payload index to delete by payload filter
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="url",
            field_schema="keyword",
        )
    except Exception as e:
        pass # Index already exists or other non-fatal error
        
    unique_urls = list(set(meta["url"] for meta in metadata if "url" in meta))
    if unique_urls:
        print(f"🗑️ Deleting old data for {len(unique_urls)} updated pages to prevent duplicates...")
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        for url in unique_urls:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="url",
                            match=MatchValue(value=url)
                        )
                    ]
                )
            )
    
    points = []
    BATCH_SIZE = 50 # Reduced from 100 to avoid triggering burst limits
    print(f"🚀 Vectorizing {len(documents)} chunks using Google Gemini Batch API...")
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_meta = metadata[i:i+BATCH_SIZE]
        
        requests_payload = [{"model": "models/gemini-embedding-2", "content": {"parts": [{"text": doc}]}} for doc in batch_docs]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={GEMINI_API_KEY}"
        
        # Exponential backoff for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            response = requests.post(url, json={"requests": requests_payload})
            if response.status_code == 200:
                embeddings = [item.get("values", []) for item in response.json().get("embeddings", [])]
                
                for j, emb in enumerate(embeddings):
                    if not emb:
                        continue
                    payload_data = {"document": batch_docs[j]}
                    payload_data.update(batch_meta[j])
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()), 
                            vector=emb, 
                            payload=payload_data
                        )
                    )
                break
            elif response.status_code == 429:
                print(f"⚠️ Rate limit hit. Google requires a cool-down. Sleeping for 60s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(60)
            else:
                raise Exception(f"HTTP Error {response.status_code} occurred while communicating with Gemini API.")
        else:
            raise Exception(f"Failed to embed batch after {max_retries} retries.")
            
        time.sleep(10) # 10 second pause between batches to completely evade rate limits
        print(f"✅ Processed {min(i+BATCH_SIZE, len(documents))}/{len(documents)} chunks...")

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