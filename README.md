# AI Knowledge Assistant

A FastAPI-based AI assistant that provides conversational chat capabilities and Retrieval-Augmented Generation (RAG) for PDF documents. Built with OpenRouter for LLM integration and FAISS for vector search.

## Features

- **Conversational Chat**: Simple chat interface powered by Mistral 7B Instruct model via OpenRouter
- **PDF Upload and Processing**: Upload PDF documents for knowledge base creation
- **RAG-Powered Q&A**: Ask questions about uploaded PDFs with context-aware responses
- **Vector Search**: Uses FAISS and sentence-transformers for efficient document retrieval
- **RESTful API**: Clean FastAPI endpoints for easy integration

## Project Structure

```
ai-knowledge-assistant/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── chat.py          # Plain chat endpoint
│   │   └── rag.py           # RAG endpoints (upload PDF, chat with PDF)
│   └── services/
│       ├── llm_service.py   # OpenRouter/OpenAI integration
│       └── rag_service.py   # PDF processing and vector search
├── data/
│   ├── faiss_index/         # Saved FAISS vector database
│   └── uploads/             # Uploaded PDF files
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/ai-knowledge-assistant
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```
   
   Get your API key from [OpenRouter](https://openrouter.ai/).

## Usage

### Running the Application

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### API Endpoints

#### Health Check
- **GET** `/`
- Returns: `{"status": "ok"}`

#### Chat
- **POST** `/chat`
- Body: `{"prompt": "Your message here"}`
- Returns: `{"response": "AI response"}`

Example:
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, how are you?"}'
```

#### Upload PDF
- **POST** `/rag/upload`
- Content-Type: `multipart/form-data`
- Form field: `file` (PDF file)
- Returns: Success message or error

Example:
```bash
curl -X POST "http://127.0.0.1:8000/rag/upload" \
     -F "file=@/path/to/your/document.pdf"
```

#### Chat with PDF
- **POST** `/rag/chat-with-pdf`
- Body: `{"prompt": "Your question about the uploaded PDF"}`
- Returns: `{"response": "Context-aware answer"}`

Example:
```bash
curl -X POST "http://127.0.0.1:8000/rag/chat-with-pdf" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is the main topic of the document?"}'
```

## How It Works

### Plain Chat
The `/chat` endpoint sends user prompts directly to the Mistral 7B Instruct model via OpenRouter, returning conversational responses.

### RAG Workflow
1. **Upload**: PDFs are uploaded via `/rag/upload`
2. **Processing**: 
   - PDF text is extracted using PyPDFLoader
   - Text is split into 500-character chunks with 50-character overlap
   - Chunks are embedded using sentence-transformers/all-MiniLM-L6-v2
   - Embeddings are stored in a FAISS vector database
3. **Querying**:
   - User questions are embedded
   - FAISS finds the 3 most similar chunks (score < 0.5 threshold)
   - Retrieved context is fed to the LLM with strict instructions
   - Response is generated based only on the provided context

## Dependencies

Key packages (see `requirements.txt` for full list):

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **OpenAI**: LLM API client (configured for OpenRouter)
- **LangChain**: Document processing and vector stores
- **FAISS**: Vector similarity search
- **Sentence-Transformers**: Text embeddings
- **PyPDFLoader**: PDF text extraction

## Configuration

### Environment Variables
- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)

### Model Configuration
- Model: `mistralai/mistral-7b-instruct-v0.1`
- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk Size: 500 characters
- Chunk Overlap: 50 characters
- Similarity Threshold: 0.5

## Development

### Running Tests
Currently no automated tests are implemented. Manual testing via API calls is recommended.

### Code Style
Follow standard Python conventions. Use type hints where possible.

## Limitations

- Only supports PDF documents
- Vector database is overwritten on each new PDF upload (no multi-document support)
- Synchronous processing (may block on large PDFs)
- Context limited to 3000 characters per query
- No user authentication or rate limiting

## Future Improvements

- Support for multiple document formats
- Persistent multi-document vector stores
- Async processing for better performance
- User sessions and conversation history
- Authentication and API rate limiting
- Frontend interface
- Advanced RAG techniques (reranking, hybrid search)

## License

This project is open-source. Feel free to use and modify as needed.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

For questions or issues, please open a GitHub issue.