from fastapi import APIRouter, Body
from app.services.llm_service import generate_response

router = APIRouter()

@router.post("/chat")
def chat(prompt: str = Body(..., embed=True)):
    response = generate_response([
        {"role": "user", "content": prompt}
    ])
    return {"response": response}