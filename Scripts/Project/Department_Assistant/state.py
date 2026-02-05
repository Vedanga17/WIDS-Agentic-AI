"""

State definition for the Department Knowledge Assistant (RAG Pipeline)
Defines the structure of data that flows through the LangGraph workflow

"""

from typing import TypedDict, List, Optional, Dict # importing dependencies; will be used for defining state

class PipelineState(TypedDict): 
    """
    State schema for the RAG pipeline
    Each node reads from and updates this state
    """
    
    # This is the SCRAPER NODE Output
    # Raw scraped content from the website (list of page contents)
    scraped_pages: Optional[List[Dict[str, str]]]  # [{"url": "...", "content": "...", "type": "web"}]
    
    # Downloaded PDF files
    pdf_files: Optional[List[Dict[str, str]]]  # [{"url": "...", "file_path": "..."}]
    
    # This is the PROCESSOR NODE Output
    # Text chunks after splitting the content
    chunks: Optional[List[str]]
    
    # Document IDs after storing in vector database
    stored_doc_ids: Optional[List[str]]
    
    # ===== USER INPUT =====
    # User's question
    query: Optional[str]
    
    # ===== RETRIEVER NODE OUTPUT =====
    # Retrieved relevant documents from vector store
    retrieved_docs: Optional[List[Dict[str, str]]]  # [{"content": "...", "metadata": {...}}]
    
    # ===== RESPONDER NODE OUTPUT =====
    # Final generated response
    response: Optional[str]
    
    # ===== METADATA =====
    # Track errors or status messages
    status: Optional[str]
    error: Optional[str]
