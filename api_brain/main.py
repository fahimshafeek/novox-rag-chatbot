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
            limit=10 
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
    # THE TWO-SHOT NUCLEAR PROMPT
    # =========================================================================
    system_prompt = f"""You are the professional AI assistant for Novox EdTech.
    Your ONLY job is to extract the answer from the Context.
    
    CRITICAL RULES:
    1. Output MUST be valid JSON with a single key "answer".
    2. Write a comprehensive, detailed, and professional paragraph (3 to 6 sentences) answering the user's question completely.
    3. NEVER use meta-phrases like "Context:", "Result:", "The text mentions", or "Paragraph". Just give the direct answer.
    4. NO bullet points, checklists, or quotation marks.
    5. NEVER include any URLs, website links, or "Click here to learn more" links in your answer.
    6. MISSING INFO: If the answer cannot be reasonably deduced from the Context, you must return exactly this: {{"answer": "I do not have that specific information available at the moment. Please contact Novox EdTech directly."}}
    
    GUIDELINES:
    - Be highly deductive. If the context strongly implies the answer (e.g., "Novox Edtech | Best Software Training"), deduce that Novox is a software training institute.
    
    EXAMPLE 1 (Found in Context):
    Context: Novox is located in Calicut.
    Question: Where is Novox?
    Output: {{"answer": "Novox EdTech is located in Calicut."}}
    
    EXAMPLE 2 (Not Found in Context):
    Context: Novox teaches Python.
    Question: Who is the CEO?
    Output: {{"answer": "I do not have that specific information available at the moment. Please contact Novox EdTech directly."}}
    
    REAL INPUT:
    Context:
    {context_string}
    """

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
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
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        response_data = response.json()
        parts = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        
        raw_text = ""
        for part in parts:
            if not part.get("thought", False):
                raw_text = part.get("text", "").strip()
                break
                
        # Fallback if no non-thought part is found
        if not raw_text and parts:
            raw_text = parts[-1].get("text", "").strip()
        
        # Parse JSON
        try:
            parsed_json = json.loads(raw_text)
            generated_answer = parsed_json.get("answer", raw_text)
        except json.JSONDecodeError:
            generated_answer = raw_text
            
        # (Removed Assassin Filter to allow for longer and detailed responses)
                
        # =========================================================================
        # THE GUILLOTINE: Instantly chop off the appended sources tail
        # =========================================================================
        generated_answer = re.split(r'(?i)(?:\\n|\n|\s)*sources?:', generated_answer)[0].strip()
        
        # =========================================================================
        # THE DECAPITATOR: Chop off hallucinated meta-prefixes at the start
        # =========================================================================
        # Strips out prefixes like "Context:", "Result:", "Answer:"
        generated_answer = re.sub(r'(?i)^(context|result|answer|paragraph \d+):\s*', '', generated_answer).strip()
        # Strips out conversational crutches like "The text mentions that"
        generated_answer = re.sub(r'(?i)^the text (mentions|says|states)( that)?\s*', '', generated_answer).strip()
        
        # Capitalize the first letter since we might have just chopped off the start of the sentence
        if generated_answer:
            generated_answer = generated_answer[0].upper() + generated_answer[1:]
        # =========================================================================
                
    except Exception as e:
        error_msg = str(e)
        if GEMINI_API_KEY:
            error_msg = error_msg.replace(GEMINI_API_KEY, "********")
        return {"error": f"LLM generation failed: {error_msg}"}
    
    unique_sources = [{"source": src} for src in list(set([item["source"] for item in retrieved_data if item["source"] != "local-test"]))] if 'retrieved_data' in locals() else []
    
    return {
        "question": request.question,
        "answer": generated_answer,
        "sources": unique_sources
    }