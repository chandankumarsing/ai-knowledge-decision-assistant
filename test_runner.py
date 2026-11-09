import os
import sys
from sqlalchemy.orm import Session
from database import SessionLocal, Organization, DocumentChunk, Document, init_db
import json

def seed_and_test():
    db = SessionLocal()
    init_db()

    # Clear prior data
    db.query(DocumentChunk).delete()
    db.query(Document).delete()
    db.query(Organization).delete()
    db.commit()

    print("🚀 Initializing Multi-Tenant Verification Sets...")
    
    # Create Tenant A (University Administration)
    org_a = Organization(name="Alpha University", email="alpha@edu.com", password_hash="hash1")
    # Create Tenant B (Corporate HR Division)
    org_b = Organization(name="Beta Corporate", email="beta@corp.com", password_hash="hash2")
    db.add_all([org_a, org_b])
    db.commit()

    # Simulate Embedding Vector data arrays
    mock_vector = json.dumps([0.1] * 1536)

    # Seed Tenant A Knowledge Rules (Merit Policy focus)
    doc_a = Document(organization_id=org_a.id, filename="Scholarship_Rules.pdf", file_size=5000, page_count=2)
    db.add(doc_a)
    db.flush()
    db.add(DocumentChunk(
        organization_id=org_a.id, document_id=doc_a.id, page_number=1,
        content="Alpha Merit Policy: Requirements dictate that candidates maintain a 3.5 GPA and be in academic year 2 or above. If a sports scholarship is currently active, additional funding caps apply.",
        embedding=mock_vector
    ))

    # Seed Tenant B Knowledge Rules (HR Compliance focus)
    doc_b = Document(organization_id=org_b.id, filename="Staff_Handbook.pdf", file_size=4000, page_count=1)
    db.add(doc_b)
    db.flush()
    db.add(DocumentChunk(
        organization_id=org_b.id, document_id=doc_b.id, page_number=1,
        content="Beta Corp Travel policy: Standard daily business mileage expense reimbursement is fixed at $0.62 per mile.",
        embedding=mock_vector
    ))
    db.commit()

    print("\n--- System Configuration Complete ---")
    print(f"Alpha Org Assistant Token: {org_a.assistant_token}")
    print(f"Beta Org Assistant Token:  {org_b.assistant_token}")
    print("--------------------------------------\n")
    
    db.close()

if __name__ == "__main__":
    seed_and_test()
