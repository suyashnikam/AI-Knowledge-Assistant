from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_DB_PATH = "data/faiss_index"

vector_db = None


def process_pdf(file_path: str):
    global vector_db

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    vector_db = FAISS.from_documents(chunks, embedding_model)

    # Save DB
    os.makedirs("data", exist_ok=True)
    vector_db.save_local(VECTOR_DB_PATH)

    return "PDF processed and stored successfully"


def load_vector_db():
    global vector_db

    if os.path.exists(VECTOR_DB_PATH):
        vector_db = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )


def query_pdf(query: str):
    global vector_db

    if vector_db is None:
        load_vector_db()

    if vector_db is None:
        return "No PDF uploaded yet."

    docs_with_scores = vector_db.similarity_search_with_score(query, k=3)

    print("Scores:", docs_with_scores)

    # 🔥 better threshold
    relevant_docs = [
        doc for doc, score in docs_with_scores if score < 1.5
    ]

    # 🔥 fallback
    if not relevant_docs:
        relevant_docs = [doc for doc, _ in docs_with_scores]

    context = "\n".join([doc.page_content for doc in relevant_docs])

    return context