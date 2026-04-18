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
    return " ".join(text.split())


# ==============================
# INGESTION
# ==============================

def process_pdf(file_path: str):
    global vector_db

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Better chunking for semantic continuity
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Attach metadata (only source)
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(file_path)

    # Load existing DB (multi-doc support)
    if vector_db is None:
        load_vector_db()

    if vector_db:
        vector_db.add_documents(chunks)
    else:
        vector_db = FAISS.from_documents(chunks, embedding_model)

    os.makedirs("data", exist_ok=True)
    vector_db.save_local(VECTOR_DB_PATH)

    return "PDF processed and stored successfully"


# ==============================
# LOAD DB
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
# QUERY
# ==============================

def query_pdf(query: str):
    global vector_db

    if vector_db is None:
        load_vector_db()

    if vector_db is None:
        return "No PDF uploaded yet.", []

    # High recall retrieval (NO filtering)
    docs_with_scores = vector_db.similarity_search_with_score(query, k=15)

    if not docs_with_scores:
        return None, []

    print("\n🔍 Similarity Scores:")
    for doc, score in docs_with_scores:
        print(
            f"Score: {score:.4f} | "
            f"Source: {doc.metadata.get('source')} | "
            f"Page: {doc.metadata.get('page')}"
        )

    # Take top-k directly (no threshold hacks)
    docs = [doc for doc, _ in docs_with_scores[:8]]

    # Build context
    context = "\n\n".join([
        f"[{doc.metadata.get('source')} | Page {doc.metadata.get('page')}]\n"
        f"{clean_text(doc.page_content)}"
        for doc in docs
    ])

    sources = list(set([
        doc.metadata.get("source")
        for doc in docs
    ]))

    return context, sources