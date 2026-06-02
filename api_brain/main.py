from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import requests
import os
import json
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Novox EdTech Brain")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Securely fetch the shiny new Gemma 4 key from Render's environment!
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
collection_name = "novox_knowledge"

print("Loading embedding model...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("Model loaded successfully!")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Novox EdTech Brain is running!"}

@app.post("/ask")
def ask_bot(request: QueryRequest):
    # Step A: Convert user question to vector
    query_vector = next(model.embed([request.question])).tolist()
    
    try:
        # Step B: Retrieve context from Qdrant Cloud
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
    # THE FIX: We explicitly demand a JSON structure to stop the CoT rambling
    system_prompt = f"""You are a precise, factual assistant for Novox EdTech. 
    Answer the user's question using ONLY the provided context. You may synthesize information from multiple chunks to form your answer. Keep it concise (1-3 sentences).
    
    CRITICAL INSTRUCTION: You must respond ONLY with a valid JSON object containing a single key "answer". Do not include any internal reasoning, scratchpads, constraints, or bullet points.
    Example format: {{"answer": "Novox Edtech is a leading IT institute based in Calicut."}}
    
    If the answer is not in the context, the "answer" value should be exactly: "I do not have that information."
    
    Context:
    {context_string}"""

    # Step E: Request generation using Google's Gemini API for Gemma 4 31B
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{"text": request.question}]
            }],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json" # THE FIX: Forces Google's API to only accept JSON output
            }
        }
        
        # Send the request directly to Google's servers
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        # Traverse the JSON response structure for the Gemini API
        response_data = response.json()
        raw_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        
        # THE FIX: Parse the guaranteed JSON to extract just the beautiful answer
        try:
            parsed_json = json.loads(raw_text)
            generated_answer = parsed_json.get("answer", raw_text)
        except json.JSONDecodeError:
            generated_answer = raw_text.strip()
        
    except Exception as e:
        return {"error": f"LLM generation failed. Is the GEMINI_API_KEY set in Render? Details: {str(e)}"}
    
    # Return the clean, AI-generated answer plus the exact web sources
    return {
        "question": request.question,
        "answer": generated_answer,
        "sources": retrieved_data
    }