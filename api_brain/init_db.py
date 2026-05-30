from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
collection_name = "edtech_knowledge"

# 1. Create the collection
if not client.collection_exists(collection_name=collection_name):
    print(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# 2. Load the model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Define and upload test data
chunks = [
    "The Python masterclass at Novox EdTech runs every weekday at 10 AM.",
    "Novox flutter students must submit their final project by week 12.",
    "Guest wifi password in the Kozhikode branch is NovoxGuest2026."
]

points = []
for i, text in enumerate(chunks):
    vector = model.encode(text).tolist()
    points.append(
        PointStruct(id=i, vector=vector, payload={"text": text})
    )

print("Uploading test vectors...")
client.upsert(
    collection_name=collection_name,
    points=points
)
print("Database initialized successfully.")