from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import requests
import os
import json
import re
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

    # Step D: Construct the Aggressive, Hardened System Prompt
    system_prompt = f"""You are a strict, factual backend API that returns a single JSON object.
    
    Task: Answer the user's question using ONLY the provided context. Keep it to 1-3 sentences.
    
    ABSOLUTE PROHIBITIONS:
    1. Do NOT repeat or quote any of the context chunks directly using quotation marks.
    2. Do NOT list constraints, scratchpads, or your internal thought process.
    3. Do NOT append any list of links, URLs, or text like "Sources:" at the bottom. The backend handles citations separately.
    
    REQUIRED OUTPUT FORMAT:
    You must output ONLY a valid JSON object matching this schema:
    {{"answer": "Your clean, direct 1-3 sentence response here."}}
    
    If the context does not contain the answer, your JSON must be exactly:
    {{"answer": "I do not have that information."}}
    
    Context data:
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
                "responseMimeType": "application/json"
            }
        }
        
        # Send the request directly to Google's servers
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        response_data = response.json()
        raw_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        
        try:
            parsed_json = json.loads(raw_text)
            generated_answer = parsed_json.get("answer", raw_text)
        except json.JSONDecodeError:
            generated_answer = raw_text.strip()
        
        # =========================================================================
        # THE FAIL-SAFE WASH: Programmatically drop quoted strings and links
        # =========================================================================
        clean_lines = []
        for line in generated_answer.split('\n'):
            l = line.strip()
            if not l:
                continue
            
            # Drop metadata or constraint markers
            if any(m in l for m in ["User question:", "Context provided:", "Constraint", "Sources:"]):
                continue
                
            # Drop any lingering raw lines that are completely wrapped in double quotes
            if (l.startswith('"') and l.endswith('"')) or (l.startswith('**"') and l.endswith('"**')):
                continue
                
            # Drop plain inline URLs if the model hallucinated any text links
            if "https://" in l or "http://" in l:
                continue
                
            clean_lines.append(l)
            
        # Re-synthesize into a clean single paragraph response
        generated_answer = " ".join(clean_lines).strip()
        # =========================================================================
        
    except Exception as e:
        return {"error": f"LLM generation failed. Is the GEMINI_API_KEY set in Render? Details: {str(e)}"}
    
    # Return the clean, AI-generated answer plus the exact web sources
    return {
        "question": request.question,
        "answer": generated_answer,
        "sources": retrieved_data
    }