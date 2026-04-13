from fastapi import FastAPI
from app.api import chat, rag

app = FastAPI(title="AI Knowledge Assistant")

app.include_router(chat.router, prefix="")
app.include_router(rag.router, prefix="")

@app.get("/")
def health():
    return {"status": "ok"}