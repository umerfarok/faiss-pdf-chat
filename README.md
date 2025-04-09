# FAISS RAG Chat Application

This application implements a Retrieval-Augmented Generation (RAG) system using FastAPI, FAISS vector database, and OpenAI's models. It allows you to chat with your PDF documents by processing them into embeddings and retrieving relevant information to answer questions.

## Features

- **PDF Processing**: Automatically processes PDFs from a specified folder
- **Vector Storage**: Uses FAISS to efficiently store and query document embeddings
- **Incremental Updates**: Processes only new or modified PDFs
- **Chunking**: Splits documents into manageable chunks with configurable overlap
- **RAG Chat**: Retrieval-augmented generation for accurate answers based on your documents
- **GPU Acceleration**: Optional GPU support for faster vector operations
- **Advanced Indexing**: Multiple FAISS index types for optimized performance

https://github.com/user-attachments/assets/78019ca1-4385-4d13-b2e4-7ff927b429c0



## Setup

1. Clone this repository
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key and other settings (see `.env.example`)
5. Place your PDF documents in the `./database/` folder

## Usage

1. Start the FastAPI server:
   ```
   python app.py
   ```
   or
   ```
   uvicorn app:app --reload
   ```

2. The API will be available at http://localhost:8000

3. Access the interactive API documentation at http://localhost:8000/docs

4. Use the web interface at http://localhost:8000 to chat with your documents

## API Endpoints

- **POST /chat**: Ask a question about your documents
- **POST /process-pdfs**: Manually trigger PDF processing
- **GET /health**: Check API health and vector store status
- **GET /**: Web interface for chatting with your documents

## Configuration

The following environment variables can be set in your `.env` file:

### Basic Configuration
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `PDF_FOLDER`: Path to your PDF documents (default: "./database/")
- `VECTOR_STORE_PATH`: Path where vector store will be saved (default: "./vectorstore/")
- `SYNC_FOLDER_DATA`: Whether to sync PDFs at startup (default: "true")

### Chunking Configuration
- `CHUNK_SIZE`: Text chunk size in tokens (default: 2000)
- `CHUNK_OVERLAP`: Overlap between text chunks (default: 200)

### Model Configuration
- `EMBEDDING_MODEL`: OpenAI embedding model to use (default: "text-embedding-3-large")
  - Options: "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"
- `CHAT_MODEL`: OpenAI chat model to use (default: "gpt-4o")
  - Options: "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"
- `MAX_TOKENS`: Maximum tokens to generate in response (default: 1000)
- `TEMPERATURE`: Temperature for response generation (0.0-1.0, default: 0.2)

### Search Configuration
- `TOP_K_RESULTS`: Number of relevant chunks to retrieve (default: 6)

### GPU Acceleration
- `USE_GPU`: Use GPU for FAISS operations if available (default: "false")
- `GPU_DEVICE_ID`: GPU device ID to use (default: 0)

### Advanced FAISS Configuration
- `FAISS_INDEX_TYPE`: Type of FAISS index to use (default: "flat")
  - Options: "flat" (most accurate), "hnsw" (balanced), "ivf" (fastest)
- `HNSW_M`: HNSW index graph connectivity (default: 32)
- `HNSW_EF_CONSTRUCTION`: HNSW index construction quality parameter (default: 200)
- `HNSW_EF_SEARCH`: HNSW index search quality parameter (default: 128)
- `IVF_NLIST`: Number of IVF clusters (default: 100)

### Performance Configuration
- `PARALLEL_PROCESSING`: Number of documents to process in parallel (default: 4)
- `EMBEDDING_BATCH_SIZE`: Number of chunks to embed in one batch (default: 32)

## Troubleshooting

### OpenAI API Errors

If you see errors like:
```
Error generating embeddings: Client.__init__() got an unexpected keyword argument 'proxies'
```

Make sure you're using the latest OpenAI Python package (version 1.3.0 or higher):
```
pip install --upgrade openai
```

### FAISS GPU Support

To use GPU acceleration:
1. Install the FAISS GPU version that matches your CUDA version
2. Set `USE_GPU=true` in your `.env` file

### Memory Issues

If you encounter memory issues with large document collections:
1. Reduce `CHUNK_SIZE` to create smaller chunks
2. Switch to a more efficient index type (e.g., "ivf" or "hnsw")
3. Process fewer documents at once by adjusting `PARALLEL_PROCESSING`
# faiss-pdf-chat
