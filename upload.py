import json
from qdrant_client import QdrantClient

# Your teammate's Qdrant URL
QDRANT_URL = "https://8aefbe5f6e3f61.lhr.life"
COLLECTION_NAME = "novox_knowledge"

print(f"🔌 Connecting to Qdrant server at {QDRANT_URL}...")
# FastEmbed is automatically triggered by the QdrantClient
client = QdrantClient(url=QDRANT_URL)

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

    print(f"🚀 Vectorizing and uploading {len(documents)} chunks to Qdrant...")
    print("(Note: It may take a moment to download the lightweight embedding model on the first run)")
    
    # client.add() automatically creates the collection, embeds the text, and uploads it!
    client.add(
        collection_name=COLLECTION_NAME,
        documents=documents,
        metadata=metadata
    )
    
    print("\n✅ Upload Complete! Your data is now live on your teammate's server.")

if __name__ == "__main__":
    process_and_upload()