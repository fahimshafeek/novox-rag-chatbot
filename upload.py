import json
import os
from qdrant_client import QdrantClient

# Your ngrok URL (ensure this is updated if your tunnel restarts)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "novox_knowledge"

print(f"🔌 Connecting to Qdrant server at {QDRANT_URL}...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# CRITICAL: Force Qdrant to use the exact same ONNX model as your FastAPI backend
client.set_model("BAAI/bge-small-en-v1.5")

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
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"🗑️ Deleting old '{COLLECTION_NAME}' collection to prevent duplicate chunks...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    print(f"🚀 Vectorizing and uploading {len(documents)} chunks to Qdrant...")
    print("(Note: It may take a moment to download the lightweight embedding model on the first run)")
    
    # client.add() automatically creates the collection, embeds the text, and uploads it!
    client.add(
        collection_name=COLLECTION_NAME,
        documents=documents,
        metadata=metadata
    )
    
    print("\n✅ Upload Complete! Your data is now live on the host server.")

if __name__ == "__main__":
    process_and_upload()