from fastapi import APIRouter, UploadFile, File, Body, HTTPException
import shutil
import os

from app.services.rag_service import process_pdf, query_pdf
from app.services.llm_service import generate_response
import uuid


router = APIRouter(prefix="/rag", tags=["RAG"])

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed"}

    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file.file.close()

    try:
        result = process_pdf(file_path)
        return {"message": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/chat-with-pdf")
def chat_with_pdf(prompt: str = Body(..., embed=True)):
    context = query_pdf(prompt)

    # Case 1: No PDF uploaded
    if context == "No PDF uploaded yet.":
        return {"response": context}

    # Case 2: No relevant context found
    if context is None:
        return {"response": "I don't know"}

    # limit context size (important)
    context = context[:3000]

    response = generate_response([
        {
            "role": "system",
            "content": f"""
You are a strict assistant.

Rules:
- Answer ONLY from the provided context
- If the answer is not clearly present, respond with: "I don't know"
- Do NOT use prior knowledge

Context:
{context}
"""
        },
        {"role": "user", "content": prompt}
    ])

    return {"response": response}