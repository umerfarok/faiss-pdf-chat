import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import shutil

import faiss
import numpy as np
from openai import OpenAI  # Updated import
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import PyPDF2
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
import tiktoken
from contextlib import asynccontextmanager
# New imports for intent classification and HuggingFace model
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Import CORS middleware
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY",""))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Configuration variables from .env
SYNC_FOLDER_DATA = os.getenv("SYNC_FOLDER_DATA", "true").lower() == "true"
PDF_FOLDER = os.getenv("PDF_FOLDER", "./database/")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vectorstore/")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "6"))
PROCESSED_FILES_PATH = os.path.join(VECTOR_STORE_PATH, "processed_files.json")

# Phishing Classifier Configuration
PHISHING_MODEL_NAME = os.getenv("PHISHING_MODEL_NAME", "Abdelllm2025/phishing_clf_fine_tuned_bert")
ENABLE_PHISHING_CLASSIFIER = os.getenv("ENABLE_PHISHING_CLASSIFIER", "true").lower() == "true"

# Advanced configuration
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "flat").lower()

# FAISS advanced parameters
IVF_NLIST = int(os.getenv("IVF_NLIST", "100"))
HNSW_M = int(os.getenv("HNSW_M", "32"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "128"))

# Performance tuning
PARALLEL_PROCESSING = int(os.getenv("PARALLEL_PROCESSING", "4"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Global variable for phishing classifier pipeline
phishing_classifier = None

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Startup code
    logger.info("Starting RAG Chat API")
    global vector_store, phishing_classifier
    
    # Initialize Vector Store
    vector_store = VectorStore()
    
    # Load Phishing Classifier Model if enabled
    if ENABLE_PHISHING_CLASSIFIER:
        try:
            logger.info(f"Loading phishing classification model: {PHISHING_MODEL_NAME}...")
            # Determine device: use GPU if available and configured, else CPU
            device = -1  # Default to CPU
            if USE_GPU and torch.cuda.is_available():
                device = GPU_DEVICE_ID
                logger.info(f"Using GPU device {device} for phishing model.")
            else:
                logger.info("Using CPU for phishing model.")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(PHISHING_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(PHISHING_MODEL_NAME)
            
            # Create pipeline
            phishing_classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            logger.info("Phishing classification model loaded successfully.")
            # Run a dummy inference to warm up the model
            try:
                _ = phishing_classifier("test")
                logger.info("Phishing model warmup successful.")
            except Exception as warmup_e:
                logger.warning(f"Phishing model warmup failed: {warmup_e}")

        except Exception as e:
            logger.error(f"Failed to load phishing model '{PHISHING_MODEL_NAME}': {e}")
            phishing_classifier = None
    else:
        logger.info("Phishing classifier is disabled.")
    
    # Process PDFs if syncing is enabled
    if SYNC_FOLDER_DATA:
        logger.info("PDF syncing is enabled. Processing PDFs...")
        process_pdfs(vector_store)
    else:
        logger.info("PDF syncing is disabled. Using existing vector store.")
        
    yield
    
    # Shutdown code
    logger.info("Shutting down RAG Chat API")
    # Clean up GPU memory if used by the classifier
    if phishing_classifier and hasattr(phishing_classifier.model, 'to'):
        try:
            phishing_classifier.model.to('cpu')
            logger.info("Moved phishing model back to CPU.")
        except Exception as e:
            logger.error(f"Error moving phishing model to CPU: {e}")
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache.")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Chat API", 
    description="API for chatting with your PDF documents and classifying phishing messages",
    lifespan=lifespan
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str

# Define specific response models for different intents
class ClassificationResponse(BaseModel):
    intent: str = "classification"
    result: str
    confidence: float
    remedies: List[str] = []
    processing_time: float

class AnalyticsResponse(BaseModel):
    intent: str = "analytics"
    answer: str
    sources: List[str]
    processing_time: float

# Union type for response
ChatResponseUnion = Union[ClassificationResponse, AnalyticsResponse]

# Helper class to manage vector store
class VectorStore:
    # ...existing VectorStore class code...
    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH):
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True, parents=True)
        self.index_file = self.vector_store_path / f"faiss_index_{FAISS_INDEX_TYPE}.index"
        self.metadata_file = self.vector_store_path / "metadata.json"
        self.index = None
        self.metadata = []
        self.load_or_create_index()

    def load_or_create_index(self):
        """Load existing index or create a new one if it doesn't exist"""
        if self.index_file.exists() and self.metadata_file.exists():
            logger.info(f"Loading existing FAISS index from {self.index_file}")
            try:
                self.index = faiss.read_index(str(self.index_file))
                
                # Use GPU if requested and available
                if USE_GPU and faiss.get_num_gpus() > 0:
                    logger.info(f"Moving index to GPU (device {GPU_DEVICE_ID})")
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, GPU_DEVICE_ID, self.index)
                    
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self._create_new_index()
        else:
            logger.info("Creating new FAISS index")
            self._create_new_index()

    def _create_new_index(self):
        """Create a new empty FAISS index"""
        # Determine embedding dimension based on model
        embedding_dim = 1536  # Default for text-embedding-ada-002 and text-embedding-3-small
        if EMBEDDING_MODEL == "text-embedding-3-large":
            embedding_dim = 3072  # text-embedding-3-large has 3072 dimensions
        
        logger.info(f"Creating index of type {FAISS_INDEX_TYPE} with dimension {embedding_dim}")
        
        try:
            # Create different index types based on configuration
            if FAISS_INDEX_TYPE == "flat":
                self.index = faiss.IndexFlatL2(embedding_dim)
                logger.info("Created Flat index")
            elif FAISS_INDEX_TYPE == "hnsw":
                # Initialize HNSW index - change to IndexHNSWFlat for better compatibility
                self.index = faiss.IndexHNSWFlat(embedding_dim, HNSW_M)
                self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
                self.index.hnsw.efSearch = HNSW_EF_SEARCH
                logger.info(f"Created HNSW index with M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}")
            elif FAISS_INDEX_TYPE == "ivf":
                # For IVF, we need a sample of vectors to train
                quantizer = faiss.IndexFlatL2(embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, IVF_NLIST)
                self.index.nprobe = min(IVF_NLIST // 4, 16)
                
                # IVF requires training - we'll initialize with an empty training set
                # and train it properly when we have vectors
                logger.info(f"Created IVF index with {IVF_NLIST} lists - note: requires training")
                
                # Create an empty training set for IVF index initialization
                empty_train = np.zeros((1, embedding_dim), dtype=np.float32)
                self.index.train(empty_train)
                logger.info("Initialized IVF index with empty training")
            else:
                # Default to flat index if unknown type specified
                logger.warning(f"Unknown index type '{FAISS_INDEX_TYPE}', defaulting to flat index")
                self.index = faiss.IndexFlatL2(embedding_dim)
            
            # Use GPU if requested and available
            if USE_GPU and faiss.get_num_gpus() > 0:
                try:
                    logger.info(f"Moving index to GPU (device {GPU_DEVICE_ID})")
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, GPU_DEVICE_ID, self.index)
                    logger.info("Successfully moved index to GPU")
                except Exception as e:
                    logger.error(f"Failed to move index to GPU: {str(e)}")
                    logger.info("Continuing with CPU index")
            
            self.metadata = []
            self.save_index()
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            # Fall back to simple flat index
            logger.info("Falling back to simple flat index")
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.metadata = []

    def add_documents(self, chunks: List[str], metadatas: List[Dict]):
        """Add document chunks and metadata to the index"""
        if not chunks:
            logger.info("No chunks to add")
            return
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        try:
            embeddings = self._get_embeddings(chunks)
            
            if embeddings:
                # Log embedding information for debugging
                logger.info(f"Generated {len(embeddings)} embeddings, first embedding dimension: {len(embeddings[0])}")
                
                # Make sure we have a valid FAISS index
                if self.index is None or not hasattr(self.index, 'ntotal'):
                    logger.error("Invalid FAISS index - recreating")
                    self._create_new_index()
                
                # Convert embeddings to correct numpy array format
                embeddings_array = np.array(embeddings, dtype=np.float32)
                logger.info(f"Prepared numpy array with shape: {embeddings_array.shape}")
                
                # Add embeddings to the index with explicit error handling
                try:
                    # If using IVF index that hasn't been properly trained
                    if FAISS_INDEX_TYPE == "ivf" and self.index.is_trained == False:
                        logger.info("Training IVF index with available vectors")
                        self.index.train(embeddings_array)
                    
                    # Add vectors to the index
                    self.index.add(embeddings_array)
                    original_len = len(self.metadata)
                    self.metadata.extend(metadatas)
                    logger.info(f"Added {len(chunks)} chunks to the index. New index size: {self.index.ntotal}, metadata length: {len(self.metadata)}")
                    
                    # Verify metadata length matches index size
                    if self.index.ntotal != len(self.metadata):
                        logger.warning(f"Index size ({self.index.ntotal}) doesn't match metadata length ({len(self.metadata)}). Adjusting...")
                        # Adjust metadata if needed
                        if self.index.ntotal > len(self.metadata):
                            # Add dummy entries to metadata
                            dummy_count = self.index.ntotal - len(self.metadata)
                            self.metadata.extend([{"source": "unknown", "chunk": -1}] * dummy_count)
                        else:
                            # Trim metadata
                            self.metadata = self.metadata[:self.index.ntotal]
                    
                    # Save index and metadata after successful addition
                    self.save_index()
                except Exception as e:
                    logger.error(f"Error adding embeddings to index: {str(e)}")
                    logger.error(f"FAISS index type: {type(self.index)}")
                    logger.error(f"Embeddings shape: {embeddings_array.shape}")
                    
                    # Try falling back to flat index
                    logger.info("Attempting to recreate index as flat index...")
                    try:
                        embedding_dim = embeddings_array.shape[1]
                        self.index = faiss.IndexFlatL2(embedding_dim)
                        self.index.add(embeddings_array)
                        self.metadata = metadatas.copy()
                        logger.info(f"Successfully created flat index with {len(chunks)} vectors")
                        self.save_index()
                    except Exception as e2:
                        logger.error(f"Failed to create fallback index: {str(e2)}")
                        raise
            else:
                logger.error("Failed to generate embeddings (empty result)")
        except Exception as e:
            logger.error(f"Failed to process chunks: {str(e)}")
            raise

    def save_index(self):
        """Save the index and metadata to disk"""
        try:
            if not self.index:
                logger.error("Cannot save index - index is None")
                return
                
            # If index is on GPU, move it back to CPU for saving
            cpu_index = self.index
            if USE_GPU and hasattr(faiss, 'index_gpu_to_cpu'):
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                except Exception as e:
                    logger.error(f"Error moving index from GPU to CPU: {str(e)}")
                    # Continue with original index
            
            # Log vector dimensions before saving for debugging
            ntotal = getattr(cpu_index, 'ntotal', 0)
            logger.info(f"Saving index with {ntotal} vectors")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            
            faiss.write_index(cpu_index, str(self.index_file))
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
            logger.info(f"Successfully saved index to {self.index_file} and metadata to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI API with retry logic"""
        try:
            embeddings = []
            # Process in batches, using configured batch size
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch_texts = texts[i:i+EMBEDDING_BATCH_SIZE]
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                # Sleep to avoid rate limits
                if i + EMBEDDING_BATCH_SIZE < len(texts):
                    time.sleep(0.5)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def search(self, query: str, top_k: int = TOP_K_RESULTS):
        """Search for similar documents using the query"""
        try:
            query_embedding = self._get_embeddings([query])[0]
            query_embedding_array = np.array([query_embedding], dtype=np.float32)
            
            # For HNSW index, we may need to adjust search parameters
            if FAISS_INDEX_TYPE == "hnsw" and hasattr(self.index, 'hnsw'):
                old_ef_search = self.index.hnsw.efSearch
                self.index.hnsw.efSearch = HNSW_EF_SEARCH
            
            distances, indices = self.index.search(query_embedding_array, k=top_k)
            
            # Restore original search parameter if needed
            if FAISS_INDEX_TYPE == "hnsw" and hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = old_ef_search
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.metadata):  # -1 means no result
                    metadata = self.metadata[idx]
                    results.append({
                        "metadata": metadata,
                        "distance": float(distance)
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            raise


# ... existing helper functions (get_processed_files, save_processed_files, num_tokens_from_string, chunk_text, read_pdf, process_pdfs) ...
def get_processed_files() -> dict:
    """Load the list of processed files from disk"""
    if os.path.exists(PROCESSED_FILES_PATH):
        try:
            with open(PROCESSED_FILES_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading processed files: {e}")
            return {}
    return {}


def save_processed_files(processed_files: dict):
    """Save the list of processed files to disk"""
    try:
        os.makedirs(os.path.dirname(PROCESSED_FILES_PATH), exist_ok=True)
        with open(PROCESSED_FILES_PATH, 'w') as f:
            json.dump(processed_files, f)
    except Exception as e:
        logger.error(f"Error saving processed files: {e}")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string"""
    encoding = tiktoken.get_encoding("cl100k_base")  # This works for most models
    num_tokens = len(encoding.encode(string))
    return num_tokens


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with a specified overlap"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    
    i = 0
    while i < len(tokens):
        # Get chunk_size tokens
        chunk_tokens = tokens[i:i + chunk_size]
        # Convert back to text
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
        # Move to next chunk, considering overlap
        i += (chunk_size - chunk_overlap)
    
    return chunks


def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file"""
    try:
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""


def process_pdfs(vector_store: VectorStore, background: bool = False):
    """Process all PDFs in the data folder and add them to the vector store"""
    pdf_folder = Path(PDF_FOLDER)
    if not pdf_folder.exists():
        logger.warning(f"PDF folder {pdf_folder} does not exist. Creating it.")
        pdf_folder.mkdir(exist_ok=True, parents=True)
        return

    # Get list of already processed files
    processed_files = get_processed_files()
    
    # Get all PDF files in the folder and subfolders
    pdf_files = list(pdf_folder.glob("**/*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_folder}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files. Checking for new files to process...")
    
    # Identify new files to process
    files_to_process = []
    for pdf_file in pdf_files:
        file_path = str(pdf_file)
        file_stat = os.stat(file_path)
        
        # Check if file was modified since last processing
        if file_path not in processed_files or file_stat.st_mtime > processed_files[file_path].get("last_modified", 0):
            files_to_process.append(pdf_file)
    
    if not files_to_process:
        logger.info("No new or modified PDF files to process")
        return
    
    logger.info(f"Processing {len(files_to_process)} new or modified PDF files")
    
    # Process each PDF file
    for pdf_file in tqdm(files_to_process, desc="Processing PDFs", disable=background):
        file_path = str(pdf_file)
        file_name = pdf_file.name
        
        try:
            # Read PDF content
            logger.info(f"Reading {file_name}")
            text = read_pdf(file_path)
            
            if not text.strip():
                logger.warning(f"No text extracted from {file_name}")
                continue
            
            # Chunk the text
            chunks = chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks from {file_name}")
            
            # Create metadata for each chunk
            metadatas = []
            for i, chunk in enumerate(chunks):
                metadatas.append({
                    "source": file_name,
                    "path": file_path,
                    "chunk": i,
                    "page_content": chunk,  # Add the actual content to metadata for retrieval
                    "processed_date": datetime.now().isoformat()
                })
            
            # Add chunks to vector store
            vector_store.add_documents(chunks, metadatas)
            
            # Update processed files record
            processed_files[file_path] = {
                "last_modified": os.stat(file_path).st_mtime,
                "processed_date": datetime.now().isoformat(),
                "chunks": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
    
    # Save the updated processed files record
    save_processed_files(processed_files)
    logger.info("PDF processing completed")


# --- Intent Classification and Phishing Detection ---
async def classify_intent(query: str) -> str:
    """
    Classify user query intent using OpenAI to determine if it's a classification request
    or an analytics/fact-based query.
    """
    try:
        logger.info(f"Using OpenAI to classify intent for query: '{query[:50]}...' (if longer)")
        
        system_message = """You are an intent classifier for a RAG system. 
        Your task is to determine if the user's query is asking for:
        1. "classification" - The user wants to classify text as phishing/spam or not
        2. "analytics" - The user is asking a general question to be answered using document knowledge

        For "classification" intent, look for:
        - Explicit requests to analyze, classify, or check if something is phishing/spam
        - Text that appears to be an email, message or suspicious content the user wants analyzed
        - Questions like "Is this phishing?" followed by message content

        For "analytics" intent, look for:
        - General questions about topics, concepts, or information
        - Requests for explanations, summaries, or information retrieval
        - Any query that doesn't explicitly ask for classification

        Respond with ONLY "classification" or "analytics" without explanation.
        """
        
        user_message = f"Classify this query: {query}"
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller model for efficiency with this task
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,  # Low temperature for more predictable results
            max_tokens=20     # We only need a single word response
        )
        
        intent = completion.choices[0].message.content.strip().lower()
        
        # Ensure we only get one of our two intended categories
        if "classification" in intent:
            logger.info("OpenAI classified intent as: classification")
            return "classification"
        else:
            # Default to analytics for any other response
            logger.info("OpenAI classified intent as: analytics")
            return "analytics"
            
    except Exception as e:
        # Fallback to rule-based classification if OpenAI fails
        logger.warning(f"Error using OpenAI for intent classification: {e}. Falling back to rule-based approach.")
        return rule_based_intent_classification(query)

def rule_based_intent_classification(query: str) -> str:
    """Fallback rule-based intent classification approach."""
    query_lower = query.strip().lower()
    # Keywords suggesting classification task
    classification_keywords = [
        "is this phishing", "is this spam", "classify this",
        "check this email", "scan this message", "analyze this text for phishing"
    ]
    
    # Check if the query starts with classification keywords
    if any(query_lower.startswith(keyword) for keyword in classification_keywords):
        return "classification"

    # Check for queries that contain classification keywords AND seem to include message content
    if any(keyword in query_lower for keyword in classification_keywords):
        # Basic check for message content (e.g., multiple lines, common email headers)
        if "\n" in query or "subject:" in query_lower or "from:" in query_lower or "http" in query_lower:
            # Check if it's just a question about classification vs. a request to classify
            if len(query.split()) > 15 or "\n" in query:  # Heuristic: longer queries or multiline likely contain text to classify
                return "classification"

    # Default to analytics/fact query
    return "analytics"

def extract_text_for_classification(query: str) -> str:
    """Extract the core text to be classified from the user query."""
    query_strip = query.strip()
    query_lower = query_strip.lower()
    
    # Check for prefixes like "Is this phishing:" followed by the text
    classification_prefixes = [
        "is this phishing:", "is this spam:", "classify this:",
        "check this email:", "scan this message:", "analyze this text for phishing:"
    ]
    
    # Check for prefix patterns
    for prefix in classification_prefixes:
        if query_lower.startswith(prefix):
            return query_strip[len(prefix):].strip()
            
    # If no prefix, look for content after a newline
    newline_index = query_strip.find('\n')
    if (newline_index != -1):
        first_line = query_strip[:newline_index].strip().lower()
        # Check if the first line seems like an instruction
        is_question = first_line.endswith('?')
        has_keyword = any(kw in first_line for kw in ["is this", "classify", "check", "scan", "analyze"])
        if is_question or has_keyword:
            potential_text = query_strip[newline_index:].strip()
            if len(potential_text) > 10:  # Ensure there's substantial text after newline
                return potential_text
                
    # Fallback: if we can't clearly extract the text to classify, use the whole query
    logger.warning("Could not reliably extract text to classify, using full query.")
    return query_strip


async def run_phishing_classifier(text: str) -> Dict[str, Any]:
    """Run the phishing classifier on the provided text."""
    global phishing_classifier
    
    if phishing_classifier is None:
        logger.error("Phishing classifier requested but not available.")
        raise HTTPException(status_code=503, detail="Phishing classification model is not available")
        
    if not text.strip():
        logger.warning("Received empty text for phishing classification.")
        return {"result": "not phishing", "confidence": 1.0}
        
    try:
        start_time = time.time()
        logger.info(f"Running phishing classification on text (length: {len(text)})")
        
        # The model might return results in different formats, handle both possibilities
        results = phishing_classifier(text, truncation=True, max_length=512)
        logger.info(f"Classification result: {results}")
        
        if isinstance(results, list) and results:
            result = results[0]  # Get the first result
            raw_label = result.get('label', '').upper()
            confidence = result.get('score', 0.0)
            
            # Map the label to a consistent format
            if raw_label == 'LABEL_1' or raw_label == 'PHISHING':
                label = "phishing"
            elif raw_label == 'LABEL_0' or raw_label == 'NOT PHISHING':
                label = "not phishing"
            else:
                logger.warning(f"Unknown label: {raw_label}, defaulting to 'not phishing'")
                label = "not phishing"
                
            logger.info(f"Classified as: {label} with confidence {confidence:.4f}")
            return {"result": label, "confidence": confidence}
        else:
            logger.error(f"Unexpected result format from classifier: {results}")
            raise HTTPException(status_code=500, detail="Unexpected classifier result format")
            
    except Exception as e:
        logger.error(f"Error during phishing classification: {e}")
        raise HTTPException(status_code=500, detail=f"Error during phishing classification: {str(e)}")


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def generate_answer(question: str, context: List[Dict]) -> str:
    """Generate an answer using OpenAI API with the given context"""
    try:
        # Extract text from context results
        context_texts = []
        seen_sources = set()
        
        for result in context:
            metadata = result.get('metadata', {})
            source = metadata.get('source', 'Unknown Source')
            content = metadata.get('page_content', '')
            
            # Include source info only once per document
            source_info = f"Source: {source}" if source not in seen_sources else ""
            if source_info:
                seen_sources.add(source)
                
            context_texts.append(f"{source_info}\n\n{content}")
            
        context_str = "\n\n---\n\n".join(context_texts)
        
        system_message = f"""You are a helpful assistant specialized in analyzing information from provided documents.
        Answer the user's question based *only* on the context provided below.
        Do not use any prior knowledge or information outside the context.
        If the answer cannot be found in the context, state that clearly.
        When information is found, cite the source document(s) mentioned in the context.
        Be concise and directly answer the question.

        Context:
        {context_str}
        """

        user_message = f"Based *only* on the provided context, answer the following question: {question}"
        
        logger.info(f"Generating answer for question: '{question}' using model {CHAT_MODEL}")
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        answer = completion.choices[0].message.content
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise

async def generate_phishing_remedies(text: str) -> List[str]:
    """Generate specific remedies based on the identified phishing text"""
    try:
        # Create a prompt to analyze the phishing message and generate targeted remedies
        system_message = """You are a cybersecurity expert. Analyze the phishing message 
        and provide 3-5 specific, actionable precautions the user should take related to this 
        specific phishing attempt. Be concise and direct. Each recommendation should be a single 
        sentence focused on a specific action. Don't include general advice unless relevant to this 
        specific message."""
        
        user_message = f"""Analyze this phishing message and provide 3-5 specific, actionable precautions:
        
        {text}
        
        Format your response as a simple list of recommendations, one per line."""
        
        logger.info("Generating phishing remedies")
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller model for efficiency
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,  # Allow some creativity in recommendations
            max_tokens=250    # Keep it concise
        )
        
        remedies_text = completion.choices[0].message.content.strip()
        
        # Process the response into a list of remedies
        remedies_list = []
        for line in remedies_text.split('\n'):
            line = line.strip()
            # Remove leading numbers, bullets or dashes
            line = line.lstrip('0123456789.-*• \t')
            if line and len(line) > 10:  # Ensure it's a substantial recommendation
                remedies_list.append(line)
        
        # If no valid remedies were found, provide default ones
        if not remedies_list:
            remedies_list = [
                "Never click on suspicious links in emails or messages.",
                "Do not share personal information or credentials in response to unsolicited messages.",
                "Contact the supposed sender through official channels to verify the message.",
                "Report the phishing attempt to your IT department or relevant authorities.",
                "Be cautious of messages creating urgency or threats to pressure you into action."
            ]
        
        logger.info(f"Generated {len(remedies_list)} phishing remedies")
        return remedies_list
        
    except Exception as e:
        logger.error(f"Error generating phishing remedies: {e}")
        # Return default remedies in case of error
        return [
            "Never click on suspicious links in emails or messages.",
            "Do not share personal information or credentials in response to unsolicited messages.",
            "Contact the supposed sender through official channels to verify the message.",
            "Report the phishing attempt to your IT department or relevant authorities.",
            "Be cautious of messages creating urgency or threats to pressure you into action."
        ]

# Initialize the vector store globally (handled in lifespan context manager)
vector_store: Optional[VectorStore] = None


# Modified Chat Endpoint with intent classification
@app.post("/chat", response_model=ChatResponseUnion)
async def chat(request: ChatRequest) -> Union[ClassificationResponse, AnalyticsResponse]:
    """
    Enhanced chat endpoint that:
    1. Classifies the query intent (classification or analytics)
    2. Routes to either phishing classifier or RAG pipeline based on intent
    3. Returns appropriate response format
    """
    start_time = time.time()
    question = request.question
    
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Ensure vector store is initialized
    if vector_store is None:
        logger.error("Vector store is not initialized")
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        # 1. Classify intent using OpenAI
        intent = await classify_intent(question)
        logger.info(f"Classified intent as: {intent}")
        
        # 2. Route based on intent
        if intent == "classification":
            if not ENABLE_PHISHING_CLASSIFIER or phishing_classifier is None:
                logger.warning("Classification intent detected but classifier is disabled")
                raise HTTPException(status_code=501, detail="Phishing classification is disabled")
                
            # Extract text to classify
            text_to_classify = extract_text_for_classification(question)
            logger.info(f"Extracted text for classification (length: {len(text_to_classify)})")
            
            # Run phishing classifier
            classification_result = await run_phishing_classifier(text_to_classify)
            
            # Generate remedies if classified as phishing
            remedies = []
            if classification_result["result"] == "phishing":
                remedies = await generate_phishing_remedies(text_to_classify)
            
            processing_time = time.time() - start_time
            
            # Return classification response with remedies
            return ClassificationResponse(
                intent=intent,
                result=classification_result["result"],
                confidence=classification_result["confidence"],
                remedies=remedies,
                processing_time=processing_time
            )
            
        elif intent == "analytics":
            # Use existing RAG pipeline
            search_results = vector_store.search(question, top_k=TOP_K_RESULTS)
            
            if not search_results:
                processing_time = time.time() - start_time
                return AnalyticsResponse(
                    intent=intent,
                    answer="I don't have relevant information to answer this question in the available documents.",
                    sources=[],
                    processing_time=processing_time
                )
            
            # Generate answer based on retrieved context
            answer = await generate_answer(question, search_results)
            
            # Extract unique source filenames
            sources = sorted(list({
                result.get("metadata", {}).get("source", "Unknown Source") 
                for result in search_results
            }))
            
            processing_time = time.time() - start_time
            
            # Return analytics response
            return AnalyticsResponse(
                intent=intent,
                answer=answer,
                sources=sources,
                processing_time=processing_time
            )
        else:
            # Should not happen with current implementation
            logger.error(f"Unknown intent: {intent}")
            raise HTTPException(status_code=500, detail="Unknown intent classification")
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/process-pdfs")
async def trigger_pdf_processing(background_tasks: BackgroundTasks):
    """Endpoint to manually trigger PDF processing"""
    background_tasks.add_task(process_pdfs, vector_store, True)
    return {"status": "PDF processing started in the background"}


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with phishing classifier status"""
    vector_store_size = 0
    vector_store_status = "unavailable"
    
    if vector_store and vector_store.index:
        vector_store_size = vector_store.index.ntotal
        vector_store_status = "ok"
    
    phishing_model_status = "disabled"
    if ENABLE_PHISHING_CLASSIFIER:
        phishing_model_status = "loaded" if phishing_classifier else "error"
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "vector_store_status": vector_store_status,
        "vector_store_size": vector_store_size,
        "pdf_sync_enabled": SYNC_FOLDER_DATA,
        "phishing_classifier_status": phishing_model_status,
        "phishing_model_name": PHISHING_MODEL_NAME if ENABLE_PHISHING_CLASSIFIER else None,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL
    }


@app.get("/")
async def read_root():
    """Serve the chat interface HTML page"""
    return FileResponse("static/chat.html")


if __name__ == "__main__":
    import uvicorn
    host = "127.0.0.1"  # For local access only
    port = 8000
    uvicorn.run(app, host=host, port=port)