from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
import os
os.makedirs("uploads", exist_ok=True)

from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pypdf import PdfReader
import shutil

load_dotenv()

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database Setup ──────────────────────────────────────────
DATABASE_URL = "sqlite:///./queries.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ── Database Model ──────────────────────────────────────────
class QueryLog(Base):
    __tablename__ = "queries"
    id         = Column(Integer, primary_key=True, index=True)
    question   = Column(Text)
    answer     = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ── Groq Client ─────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Request Model ────────────────────────────────────────────
class PromptRequest(BaseModel):
    query: str

# ── API 1: Ask AI ────────────────────────────────────────────
@app.post("/ask-ai")
def ask_ai(data: PromptRequest):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": data.query}],
            max_tokens=200
        )
        answer = response.choices[0].message.content

        db = SessionLocal()
        db.add(QueryLog(question=data.query, answer=answer))
        db.commit()
        db.close()

        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}

# ── API 2: Resume Analyzer ───────────────────────────────────
@app.post("/analyze-resume")
def analyze_resume(file: UploadFile = File(...)):
    try:
        
        # Step 1 — Save uploaded PDF
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2 — Extract text from PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        if not text.strip():
            return {"error": "Could not extract text from PDF"}

        # Step 3 — Send to Groq AI
        prompt = f"""
        You are an expert resume reviewer.
        Analyze the following resume and provide:
        1. Strengths
        2. Weaknesses
        3. Improvements
        4. Overall Score out of 10

        Resume:
        {text[:3000]}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        analysis = response.choices[0].message.content

        # Step 4 — Save to DB
        db = SessionLocal()
        db.add(QueryLog(
            question=f"Resume analysis: {file.filename}",
            answer=analysis
        ))
        db.commit()
        db.close()

        return {
            "filename": file.filename,
            "analysis": analysis
        }

    except Exception as e:
        return {"error": str(e)}

# ── API 3: History ───────────────────────────────────────────
@app.get("/history")
def get_history():
    db = SessionLocal()
    logs = db.query(QueryLog).order_by(QueryLog.created_at.desc()).all()
    db.close()
    return {
        "total": len(logs),
        "history": [
            {
                "id": log.id,
                "question": log.question,
                "answer": log.answer,
                "created_at": log.created_at
            }
            for log in logs
        ]
    }

# ── API 4: Delete ────────────────────────────────────────────
@app.delete("/history/{id}")
def delete_query(id: int):
    db = SessionLocal()
    log = db.query(QueryLog).filter(QueryLog.id == id).first()
    if not log:
        return {"error": "Query not found"}
    db.delete(log)
    db.commit()
    db.close()
    return {"message": f"Query {id} deleted successfully"}

# ── API 5: First API ─────────────────────────────────────────
@app.get("/first-api")
def first_api(name: str):
    return {"message": f"Hello {name}, I am the first API!"}