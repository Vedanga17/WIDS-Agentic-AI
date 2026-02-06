"""
Configuration file for Department Assistant RAG Pipeline
Stores all settings and constants used across the project
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Get the directory where this config file is located
BASE_DIR = Path(__file__).parent.resolve()

# ========== API KEYS (from .env file) ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ========== SCRAPER SETTINGS ==========
# Base URL of the website to scrape
BASE_URL = "https://www.che.iitb.ac.in/"

# Domain to stay within (don't scrape external links)
ALLOWED_DOMAIN = "che.iitb.ac.in"

# Time delay between requests (in seconds) - be polite to the server
SCRAPER_DELAY = 1.5

# Maximum number of pages to scrape (safety limit)
MAX_PAGES = 300

# Timeout for each request (in seconds)
REQUEST_TIMEOUT = 10

# ========== CHUNKING SETTINGS ==========
# Size of each text chunk (in characters)
# Increased to 1500 to keep more context together
CHUNK_SIZE = 1500

# Overlap between chunks (helps maintain context)
# Increased to 300 for better context preservation
CHUNK_OVERLAP = 300

# ========== EMBEDDING SETTINGS ==========
# Model to use for generating embeddings (HuggingFace)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ========== VECTOR STORE SETTINGS ==========
# Path to store the Chroma database (using absolute path)
VECTOR_DB_PATH = str(BASE_DIR / "department_vector_db")

# Collection name in Chroma
COLLECTION_NAME = "che_department"

# ========== LLM SETTINGS ==========
# Model for generating responses (Groq)
LLM_MODEL = "llama-3.3-70b-versatile"

# Temperature for response generation (0-1, lower = more focused)
LLM_TEMPERATURE = 0

# Maximum tokens in response
MAX_TOKENS = 500
