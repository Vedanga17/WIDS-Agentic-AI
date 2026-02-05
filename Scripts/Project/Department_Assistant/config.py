"""
Configuration file for Department Assistant RAG Pipeline
Stores all settings and constants used across the project
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
CHUNK_SIZE = 1000

# Overlap between chunks (helps maintain context)
CHUNK_OVERLAP = 200

# ========== EMBEDDING SETTINGS ==========
# Model to use for generating embeddings (HuggingFace)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ========== VECTOR STORE SETTINGS ==========
# Path to store the Chroma database
VECTOR_DB_PATH = "./department_vector_db"

# Collection name in Chroma
COLLECTION_NAME = "che_department"

# ========== LLM SETTINGS ==========
# Model for generating responses (Groq)
LLM_MODEL = "llama-3.3-70b-versatile"

# Temperature for response generation (0-1, lower = more focused)
LLM_TEMPERATURE = 0

# Maximum tokens in response
MAX_TOKENS = 500
