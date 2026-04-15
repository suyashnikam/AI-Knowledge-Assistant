from fastapi import APIRouter, Body
from app.services.llm_service import generate_response

router = APIRouter()

@router.post("/chat")
def chat(prompt: str = Body(..., embed=True)):
    try:
        response = generate_response([...])
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}