from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests

app = FastAPI(title="Novox EdTech Brain")

client = QdrantClient(url="http://localhost:6333")
collection_name = "edtech_knowledge"

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_bot(request: QueryRequest):
    # Step A: Convert user question to vector
    query_vector = model.encode(request.question).tolist()
    
    try:
        # Step B: Retrieve context from Qdrant
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3 
        ).points
        
        # Step C: Enterprise Citation Upgrade (Grabs text AND source url)
        # We use .get("url", "local-test") just in case your dummy data lacks a URL
        retrieved_data = [{"text": hit.payload["text"], "source": hit.payload.get("url", "local-test")} for hit in search_results]
        
        # Combine just the text for the LLM to read
        context_string = "\n".join([item["text"] for item in retrieved_data])
        
    except Exception as e:
        return {"error": f"Database search failed: {str(e)}"}

    # Step D: Construct the Strict System Prompt
    system_prompt = f"""You are a precise, factual assistant for Novox EdTech. 
    Answer the user's question using ONLY the provided context. Keep your answer strictly to one sentence.
    If the answer is not in the context, reply exactly with: "I do not have that information."
    
    Context:
    {context_string}"""

    # Step E: Request generation using Ollama's CHAT endpoint
    try:
        ollama_response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.question}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            }
        )
        ollama_response.raise_for_status()
        
        # The JSON response structure is different for the Chat API
        generated_answer = ollama_response.json().get("message", {}).get("content", "")
        
    except Exception as e:
        return {"error": f"LLM generation failed: {str(e)}"}
    
    # Return the clean answer plus the exact data sources
    return {
        "question": request.question,
        "answer": generated_answer.strip(),
        "sources": retrieved_data
    }