import os
import uuid
from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# MySQL कनेक्शन स्ट्रिंग (Username, Password और Database Name अपने हिसाब से बदलें)
# हमने Driver के लिए 'pymysql' का उपयोग किया है
# DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:password@localhost:3306/ai_tenant_db")
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/ai_tenant_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Organization(Base):
    __tablename__ = "organizations"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    assistant_token = Column(String(64), unique=True, default=lambda: uuid.uuid4().hex)

class Document(Base):
    __tablename__ = "documents"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=False)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    
    # MySQL में Vector को LONGTEXT के रूप में JSON स्ट्रिंग बनाकर स्टोर करेंगे
    embedding = Column(Text(4294967295), nullable=False) 

def init_db():
    # यह आपके MySQL में टेबल्स बना देगा
    Base.metadata.create_all(bind=engine)
