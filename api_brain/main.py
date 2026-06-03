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
    # THE TWO-SHOT NUCLEAR PROMPT
    # =========================================================================
    system_prompt = f"""You are the professional AI assistant for Novox EdTech.
    Your ONLY job is to extract the answer from the Context.
    
    CRITICAL RULES:
    1. Output MUST be valid JSON with a single key "answer".
    2. Write exactly 1 to 3 natural, professional sentences.
    3. NEVER use meta-phrases like "Context:", "Result:", "The text mentions", or "Paragraph". Just give the direct answer.
    4. NO bullet points, checklists, or quotation marks.
    5. MISSING INFO: If the answer is NOT in the Context, you must return exactly this: {{"answer": "I do not have that specific information available at the moment. Please contact Novox EdTech directly."}}
    
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
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent"
        
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
            chunks = [c.strip() for c in generated_answer.split('*') if c.strip()]
            valid_chunks = []
            
            for c in chunks:
                if c.endswith("Yes.") or "?" in c or "Valid" in c or "Schema" in c:
                    continue
                if c.startswith('"') or c.endswith('"'):
                    continue
                if "1-3 sentences" in c or "quotes" in c or "links" in c:
                    continue
                    
                valid_chunks.append(c)
                
            if valid_chunks:
                generated_answer = max(valid_chunks, key=len)
                
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
    
    return {
        "question": request.question,
        "answer": generated_answer,
        "sources": []
    }