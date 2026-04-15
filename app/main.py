from fastapi import FastAPI
from app.api import chat, rag
from app.services.rag_service import load_vector_db
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting AI Knowledge Assistant...")
    print("Loading vector DB...")
    load_vector_db()

    yield  # App runs here

    # Shutdown logic (optional)
    print("Shutting down...")

app = FastAPI(title="AI Knowledge Assistant", lifespan=lifespan)

app.include_router(chat.router, prefix="/v1")
app.include_router(rag.router, prefix="/v1")

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "AI Knowledge Assistant",
        "version": "v1"
    }