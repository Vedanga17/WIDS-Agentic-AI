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
MAX_PAGES = 350

# Timeout for each request (in seconds)
REQUEST_TIMEOUT = 10

# Max retries for a transient failure (timeout / connection reset / 5xx) before giving up on a page
SCRAPER_MAX_RETRIES = 2

# ========== CHUNKING SETTINGS ==========
# Size of each text chunk (in characters)
CHUNK_SIZE = 1500

# Overlap between chunks (helps maintain context)
CHUNK_OVERLAP = 300

# ========== EMBEDDING SETTINGS ==========
# Model to use for generating embeddings (HuggingFace)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ========== VECTOR STORE SETTINGS ==========
# Path to store the Chroma database (using absolute path)
# NOTE: deliberately NOT under BASE_DIR (which lives inside OneDrive) - Chroma's
# newer engine does file locking/mmap that clashes with OneDrive's sync client
# and threw disk I/O errors when the DB lived in the synced folder. Moved out to
# a plain local path instead.
VECTOR_DB_PATH = r"C:\department_vector_db"

# Collection name in Chroma
COLLECTION_NAME = "che_department"

# ========== LLM SETTINGS ==========
# Model for generating responses (Groq)
# NOTE: llama-3.3-70b-versatile was deprecated/decommissioned by Groq (shutdown 08/16/26) -
# switched to its recommended replacement.
LLM_MODEL = "openai/gpt-oss-120b"

# Temperature for response generation (0-1, lower = more focused)
LLM_TEMPERATURE = 0

# Maximum tokens in response
MAX_TOKENS = 1500

# ========== CONVERSATION MEMORY SETTINGS ==========
# How many previous user/assistant exchanges to feed back into the responder as
# conversation history, so follow-up questions ("what about the other one?") can
# be resolved. Kept small on purpose - this is just enough for the LLM to follow
# the thread of a conversation, not a substitute for the retrieved context, which
# is still what every factual claim has to come from.
CHAT_HISTORY_TURNS = 3
