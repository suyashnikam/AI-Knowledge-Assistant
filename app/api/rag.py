from fastapi import APIRouter, UploadFile, File, Body, HTTPException
import shutil
import os
import uuid

from app.services.rag_service import process_pdf, query_pdf
from app.services.llm_service import generate_response

router = APIRouter(prefix="/rag", tags=["RAG"])

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file.file.close()

    result = process_pdf(file_path)
    return {"message": result}


@router.post("/chat-with-pdf")
def chat_with_pdf(prompt: str = Body(..., embed=True)):
    context, sources = query_pdf(prompt)

    if context == "No PDF uploaded yet.":
        return {"response": context}

    if context is None:
        return {"response": "I don't know"}

    context = context[:6000]

    response = generate_response([
        {
            "role": "system",
            "content": f"""
You are an expert information extraction assistant.

Instructions:
- Answer strictly from the context
- Extract complete information if multiple items exist
- Do not miss relevant details
- Do not guess or add external knowledge
- If answer is not present, say "I don't know"

Context:
{context}
"""
        },
        {"role": "user", "content": prompt}
    ])

    return {
        "response": response,
        "sources": sources
    }