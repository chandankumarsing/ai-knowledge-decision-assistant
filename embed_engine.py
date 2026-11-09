import pypdf
import io
import json
import os
from openai import OpenAI
from database import Document, DocumentChunk

# Fallback directly to string if the environment variable isn't active
API_KEY = os.getenv("OPENAI_API_KEY", "your-actual-openai-api-key")
client = OpenAI(api_key=API_KEY)
def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text.replace("\n", " ")]
    )
    return response.data[0].embedding
# Upgraded processing function inside embed_engine.py
def process_and_chunk_pdf(db_session, org_id: str, filename: str, file_bytes: bytes) -> dict:
    file_size = len(file_bytes)
    
    if file_size > 10 * 1024 * 1024:
        return {"success": False, "error": "File size exceeds mandatory 10MB limit."}
        
    existing_docs = db_session.query(Document).filter_by(organization_id=org_id).count()
    if existing_docs >= 10:
        return {"success": False, "error": "Knowledge base full. Max 10 documents allowed."}

    pdf_file = io.BytesIO(file_bytes)
    reader = pypdf.PdfReader(pdf_file)
    page_count = len(reader.pages)
    
    if page_count > 20:
        return {"success": False, "error": "Document length exceeds mandatory 20-page limit."}

    new_doc = Document(organization_id=org_id, filename=filename, file_size=file_size, page_count=page_count)
    db_session.add(new_doc)
    db_session.flush()

    # Python 3.14 Safe Text Aggregation Loop
    for page_idx in range(page_count):
        try:
            page = reader.pages[page_idx]
            text = page.extract_text() or ""
        except Exception:
            # Fallback path if standard structural extraction hits layout problems
            text = ""
            
        if not text.strip():
            continue
            
        # Standard window chunking layout
        chunks = [text[i:i+700] for i in range(0, len(text), 600)]
        for chunk_text in chunks:
            vector_repr = get_embedding(chunk_text)
            db_chunk = DocumentChunk(
                organization_id=org_id,
                document_id=new_doc.id,
                content=chunk_text,
                page_number=page_idx + 1,
                embedding=json.dumps(vector_repr)
            )
            db_session.add(db_chunk)
            
    db_session.commit()
    return {"success": True, "document_id": str(new_doc.id)}
