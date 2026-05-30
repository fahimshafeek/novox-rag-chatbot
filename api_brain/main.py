from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import requests

app = FastAPI(title="Novox EdTech Brain")

client = QdrantClient(url="http://localhost:6333")
collection_name = "novox_knowledge"

print("Loading embedding model...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("Model loaded successfully!")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_bot(request: QueryRequest):
    # Step A: Convert user question to vector
    query_vector = next(model.embed([request.question])).tolist()
    
    try:
        # Step B: Retrieve context from Qdrant
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="fast-bge-small-en-v1.5",
            limit=5 
        ).points
        
        # Step C: Enterprise Citation Upgrade (Grabs text AND source url safely)
        retrieved_data = []
        for hit in search_results:
            payload = hit.payload or {}
            
            # Safe extraction fallback ladder for text contents
            text_chunk = payload.get("document") or payload.get("text") or payload.get("content") or ""
            # Safe extraction fallback ladder for source links
            source_url = payload.get("url") or payload.get("source") or "local-test"
            
            if text_chunk:
                retrieved_data.append({
                    "text": text_chunk, 
                    "source": source_url
                })
        
        # Combine just the text for the LLM to read
        context_string = "\n".join([item["text"] for item in retrieved_data])
        
    except Exception as e:
        return {"error": f"Database search failed: {str(e)}"}

    # Step D: Construct the Strict System Prompt
    system_prompt = f"""You are a precise, factual assistant for Novox EdTech. 
    Answer the user's question using ONLY the provided context. You may synthesize information from multiple chunks to form your answer. Keep it concise (1-3 sentences).
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