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
    query_vector = next(model.embed([request.question])).tolist()
    
    try:
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="fast-bge-small-en-v1.5",
            limit=5 
        ).points
        
        retrieved_data = []
        for hit in search_results:
            payload = hit.payload or {}
            text_chunk = payload.get("document") or payload.get("text") or payload.get("content") or ""
            source_url = payload.get("url") or payload.get("source") or "local-test"
            
            if text_chunk:
                retrieved_data.append({"text": text_chunk, "source": source_url})
        
        context_string = "\n".join([item["text"] for item in retrieved_data])
        
    except Exception as e:
        return {"error": f"Database search failed: {str(e)}"}

    # =========================================================================
    # THE FEW-SHOT NUCLEAR PROMPT
    # =========================================================================
    system_prompt = f"""You are an AI assistant for Novox EdTech.
    Your ONLY job is to extract the answer from the Context and return it.
    
    RULES:
    1. Write exactly 1 to 3 normal sentences.
    2. NO bullet points. NO checklists. NO quotation marks.
    3. Output raw JSON format.
    
    EXAMPLE INPUT:
    Context: Novox is located in Calicut and teaches Python.
    Question: Where is Novox?
    
    EXAMPLE EXACT OUTPUT:
    {{"answer": "Novox is located in Calicut and offers courses in Python."}}
    
    REAL INPUT:
    Context:
    {context_string}
    """

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
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        response_data = response.json()
        raw_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        
        # Parse JSON
        try:
            parsed_json = json.loads(raw_text)
            generated_answer = parsed_json.get("answer", raw_text)
        except json.JSONDecodeError:
            generated_answer = raw_text
            
        # =========================================================================
        # THE ASSASSIN FILTER: Kill the bullet points if the LLM still disobeys
        # =========================================================================
        if "*" in generated_answer:
            # Split the string by the asterisk bullet points
            chunks = [c.strip() for c in generated_answer.split('*') if c.strip()]
            valid_chunks = []
            
            for c in chunks:
                # Kill checklist confirmations (e.g., "JSON format? Yes.")
                if c.endswith("Yes.") or "?" in c or "Valid" in c or "Schema" in c:
                    continue
                # Kill direct quotes wrapped in quotation marks
                if c.startswith('"') or c.endswith('"'):
                    continue
                # Kill literal constraint repeats
                if "1-3 sentences" in c or "quotes" in c or "links" in c:
                    continue
                    
                valid_chunks.append(c)
                
            # The actual answer is the longest surviving normal paragraph
            if valid_chunks:
                generated_answer = max(valid_chunks, key=len)
        # =========================================================================
                
    except Exception as e:
        return {"error": f"LLM generation failed: {str(e)}"}
    
    return {
        "question": request.question,
        "answer": generated_answer,
        "sources": retrieved_data
    }