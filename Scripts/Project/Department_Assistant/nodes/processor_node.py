"""
Processor Node - Chunking and Embedding Component
This node takes scraped text, splits it into chunks, embeds them, and stores in vector DB
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict

from config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    EMBEDDING_MODEL, 
    VECTOR_DB_PATH, 
    COLLECTION_NAME
)
from state import PipelineState


def processor_node(state: PipelineState) -> PipelineState:
    """
    Main processor node function
    Chunks the scraped text, generates embeddings, and stores in vector DB
    
    Args:
        state: Current pipeline state with scraped_pages
        
    Returns:
        Updated state with chunks and stored_doc_ids
    """
    print("⚙️  Starting text processing...")
    
    try:
        # Get scraped pages and PDFs from state
        scraped_pages = state.get("scraped_pages", [])
        pdf_files = state.get("pdf_files", [])
        
        if not scraped_pages and not pdf_files:
            raise ValueError("No scraped pages or PDFs found in state")
        
        print(f"📄 Processing {len(scraped_pages)} pages and {len(pdf_files)} PDFs...")
        
        # Step 1: Convert scraped pages to LangChain Documents
        documents = create_documents(scraped_pages)
        print(f"✅ Created {len(documents)} documents from web pages")
        
        # Step 2: Process PDFs and add to documents
        if pdf_files:
            pdf_documents = process_pdfs(pdf_files)
            documents.extend(pdf_documents)
            print(f"✅ Added {len(pdf_documents)} documents from PDFs")
        
        # Step 2: Chunk the documents
        chunks = chunk_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 3: Create embeddings and store in vector DB
        doc_ids = store_in_vectordb(chunks)
        print(f"✅ Stored {len(doc_ids)} chunks in vector database")
        
        # Update state
        state["chunks"] = [chunk.page_content for chunk in chunks]
        state["stored_doc_ids"] = doc_ids
        state["status"] = f"Successfully processed {len(chunks)} chunks"
        
        print(f"✅ Processing complete!")
        
    except Exception as e:
        state["error"] = f"Processing failed: {str(e)}"
        state["status"] = "Failed"
        print(f"❌ Processing error: {str(e)}")
    
    return state


def create_documents(scraped_pages: List[Dict[str, str]]) -> List[Document]:
    """
    Convert scraped pages into LangChain Document objects
    
    Each Document has:
    - page_content: The actual text
    - metadata: Information about the source (URL, etc.)
    
    Args:
        scraped_pages: List of dicts with 'url' and 'content' keys
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    for page in scraped_pages:
        # Create a Document with content and metadata
        doc = Document(
            page_content=page["content"],
            metadata={
                "source": page["url"],
                "type": "web_page"
            }
        )
        documents.append(doc)
    
    return documents


def process_pdfs(pdf_files: List[Dict[str, str]]) -> List[Document]:
    """
    Process downloaded PDF files and extract text
    
    Args:
        pdf_files: List of dicts with 'url' and 'file_path' keys
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    for pdf_file in pdf_files:
        try:
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(pdf_file["file_path"])
            pages = loader.load()
            
            # Add metadata to each page
            for page in pages:
                page.metadata["source"] = pdf_file["url"]
                page.metadata["type"] = "pdf"
                page.metadata["file_path"] = pdf_file["file_path"]
                documents.append(page)
            
            print(f"  ✓ Processed PDF: {pdf_file['url']} ({len(pages)} pages)")
            
        except Exception as e:
            print(f"  ⚠️  Error processing PDF {pdf_file['url']}: {e}")
    
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for better embedding and retrieval
    
    Why chunk?
    - Embeddings work better on smaller, focused text segments
    - Retrieval is more precise when chunks are topic-specific
    - LLMs have context limits, smaller chunks fit better
    
    Args:
        documents: List of LangChain Documents
        
    Returns:
        List of chunked Documents (each chunk is a new Document)
    """
    # Initialize text splitter with settings from config
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # Max characters per chunk
        chunk_overlap=CHUNK_OVERLAP,  # Overlap helps maintain context across chunks
        length_function=len,           # How to measure chunk size
        is_separator_regex=False,      # Use simple string separators
    )
    
    # Split all documents into chunks
    # This preserves metadata from original documents
    chunks = text_splitter.split_documents(documents)
    
    return chunks


def store_in_vectordb(chunks: List[Document]) -> List[str]:
    """
    Generate embeddings for chunks and store them in Chroma vector database
    
    Process:
    1. Initialize embedding model (HuggingFace)
    2. Create/load Chroma database
    3. Add documents (auto-generates embeddings)
    4. Return document IDs
    
    Args:
        chunks: List of Document chunks to embed and store
        
    Returns:
        List of document IDs stored in the database
    """
    print("🧠 Initializing embedding model...")
    
    # Initialize HuggingFace embeddings
    # This downloads the model on first use (cached afterwards)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
    )
    
    print("💾 Storing in vector database...")
    
    # Create or connect to existing Chroma database
    # If collection exists, it will be loaded; otherwise, created
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    # Add documents to vector store
    # This automatically:
    # 1. Generates embeddings for each chunk
    # 2. Stores embeddings and metadata in Chroma
    # 3. Returns list of document IDs
    doc_ids = vectorstore.add_documents(chunks)
    
    return doc_ids


# For testing the processor independently
if __name__ == "__main__":
    print("Testing processor node...")
    
    # Create sample scraped data for testing
    test_scraped_pages = [
        {
            "url": "https://www.che.iitb.ac.in/test1",
            "content": "This is a test page about chemical engineering. " * 50  # Long text
        },
        {
            "url": "https://www.che.iitb.ac.in/test2",
            "content": "This is another test page about faculty members. " * 50
        }
    ]
    
    # Create initial state
    test_state: PipelineState = {
        "scraped_pages": test_scraped_pages,
        "chunks": None,
        "stored_doc_ids": None,
        "query": None,
        "retrieved_docs": None,
        "response": None,
        "status": None,
        "error": None
    }
    
    # Run processor
    result_state = processor_node(test_state)
    
    # Print results
    if result_state.get("chunks"):
        print(f"\n✅ Processing successful!")
        print(f"Total chunks: {len(result_state['chunks'])}")
        print(f"Stored document IDs: {len(result_state['stored_doc_ids'])}")
        print(f"\nFirst chunk preview (200 chars):")
        print(result_state["chunks"][0][:200] + "...")
    else:
        print(f"\n❌ Processing failed: {result_state.get('error')}")
