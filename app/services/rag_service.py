from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

# ==============================
# CONFIG
# ==============================

VECTOR_DB_PATH = "data/faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = None


# ==============================
# UTILS
# ==============================

def clean_text(text: str) -> str:
    """Remove extra spaces/newlines from PDF text"""
    return " ".join(text.split())


# ==============================
# INGESTION (UPLOAD PDF)
# ==============================

def process_pdf(file_path: str):
    global vector_db

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Create vector DB
    vector_db = FAISS.from_documents(chunks, embedding_model)

    # Persist DB
    os.makedirs("data", exist_ok=True)
    vector_db.save_local(VECTOR_DB_PATH)

    return "PDF processed and stored successfully"


# ==============================
# LOAD EXISTING DB
# ==============================

def load_vector_db():
    global vector_db

    if os.path.exists(VECTOR_DB_PATH):
        vector_db = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("✅ Vector DB loaded")
    else:
        print("⚠️ No existing vector DB found")


# ==============================
# QUERY (RAG)
# ==============================

def query_pdf(query: str):
    global vector_db

    # If DB not loaded (fallback safety)
    if vector_db is None:
        load_vector_db()

    if vector_db is None:
        return "No PDF uploaded yet."

    # Retrieve similar docs
    docs_with_scores = vector_db.similarity_search_with_score(query, k=3)

    if not docs_with_scores:
        return None

    # Debug logs
    print("\n🔍 Similarity Scores:")
    for doc, score in docs_with_scores:
        print(f"Score: {score:.4f} | Page: {doc.metadata.get('page')}")

    # ==============================
    # DYNAMIC THRESHOLD
    # ==============================
    top_score = docs_with_scores[0][1]

    relevant_docs = [
        doc for doc, score in docs_with_scores if score <= top_score + 0.3
    ]

    # Fallback (if strict filter removes everything)
    if not relevant_docs:
        relevant_docs = [doc for doc, _ in docs_with_scores]

    # Limit number of chunks (avoid token overflow)
    relevant_docs = relevant_docs[:2]

    # Build clean context
    context = "\n\n".join([
        f"[Page {doc.metadata.get('page')}]\n{clean_text(doc.page_content)}"
        for doc in relevant_docs
    ])

    return context