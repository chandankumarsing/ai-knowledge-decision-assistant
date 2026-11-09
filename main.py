import os
import json
import math
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, Organization, DocumentChunk, init_db
from embed_engine import process_and_chunk_pdf, get_embedding
from decision_engine import run_pipeline

# 1. Initialize the FastAPI app instance first (Fixes the NameError!)
app = FastAPI(title="Multi-Tenant AI Knowledge Assistant")

# 2. Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Database session dependency injector
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 4. Request validation definitions
class RegistrationPayload(BaseModel):
    name: str
    email: str
    password: str

class QueryPayload(BaseModel):
    token: str
    question: str

# 5. Database Startup Lifecycle Hook
@app.on_event("startup")
def startup_event():
    init_db()

# 6. Admin API: Register Organization
@app.post("/api/admin/register")
def register_organization(payload: RegistrationPayload, db: Session = Depends(get_db)):
    existing = db.query(Organization).filter_by(email=payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Organization already registered.")
    
    org = Organization(name=payload.name, email=payload.email, password_hash=payload.password)
    db.add(org)
    db.commit()
    db.refresh(org)
    return {"status": "success", "assistant_token": org.assistant_token}

# 7. Admin API: Upload Knowledge Base Document
@app.post("/api/admin/upload")
async def upload_knowledge_document(
    org_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    contents = await file.read()
    result = process_and_chunk_pdf(db, org_id, file.filename, contents)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

# 8. Public API: Ask Assistant Pipeline (Enforces Strict Isolation)
@app.post("/api/ask")
def ask_assistant(payload: QueryPayload, db: Session = Depends(get_db)):
    # Look up the organization using the public token
    org = db.query(Organization).filter_by(assistant_token=payload.token).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Invalid token.")

    # Calculate embedding vector for the incoming user query
    q_vec = get_embedding(payload.question)

    # Isolated Tenant Context Filter
    all_chunks = db.query(DocumentChunk).filter_by(organization_id=org.id).all()
    
    def cosine_similarity(v1, v2):
        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm_a = math.sqrt(sum(x * x for x in v1))
        norm_b = math.sqrt(sum(x * x for x in v2))
        return dot_product / (norm_a * norm_b) if (norm_a * norm_b) else 0.0

    ranked_chunks = []
    for chunk in all_chunks:
        c_vec = json.loads(chunk.embedding)
        score = cosine_similarity(q_vec, c_vec)
        if score > 0.35:  # Relevance criteria boundary cutoff
            ranked_chunks.append((score, chunk))

    # Sort descending by relevance score and extract top 4 text elements
    ranked_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [item[1] for item in ranked_chunks[:4]]

    # Forward contextual frames to your dual LLM + System 1 engine script
    output = run_pipeline(payload.question, top_chunks)
    return output
from fastapi.responses import HTMLResponse

# Add a simple root endpoint to display a basic front-end dashboard
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Multi-Tenant Assistant Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6f9; color: #333; margin: 40px; }
            .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); max-width: 600px; margin: 0 auto; }
            h2 { color: #1e3a8a; margin-top: 0; }
            p { line-height: 1.6; color: #4b5563; }
            .status { display: inline-block; padding: 6px 12px; background: #d1fae5; color: #065f46; border-radius: 20px; font-weight: bold; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🚀 Multi-Tenant Assistant Backend Operational</h2>
            <p>Your API engine is running successfully with MySQL database connectivity.</p>
            <div class="status">● Server Online</div>
            <p>To test endpoints or view documentation, visit <a href="/docs" style="color: #2563eb; font-weight: bold;">/docs (Swagger UI)</a>.</p>
        </div>
    </body>
    </html>
    """
