"""
Retriever Node - Vector Search Component
This node retrieves relevant documents from the vector database based on user query
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import EMBEDDING_MODEL, VECTOR_DB_PATH, COLLECTION_NAME
from state import PipelineState


def retriever_node(state: PipelineState) -> PipelineState:
    """
    Main retriever node function
    Searches vector database for relevant documents based on user query
    
    Args:
        state: Current pipeline state with query
        
    Returns:
        Updated state with retrieved_docs
    """
    print("🔍 Starting document retrieval...")
    
    try:
        # Get query from state
        query = state.get("query")
        
        if not query:
            raise ValueError("No query found in state")
        
        print(f"🔎 Query: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = retrieve_documents(query)
        
        print(f"✅ Retrieved {len(retrieved_docs)} relevant documents")
        
        # Update state
        state["retrieved_docs"] = retrieved_docs
        state["status"] = f"Successfully retrieved {len(retrieved_docs)} documents"
        
    except Exception as e:
        state["error"] = f"Retrieval failed: {str(e)}"
        state["status"] = "Failed"
        print(f"❌ Retrieval error: {str(e)}")
    
    return state


def retrieve_documents(query: str, k: int = 10):
    """
    Retrieve relevant documents from vector database
    
    Process:
    1. Initialize embeddings (same model used for storage)
    2. Connect to Chroma database
    3. Convert query to embedding
    4. Find top-k similar documents using MMR for diversity
    5. Return documents with metadata
    
    Args:
        query: User's question
        k: Number of documents to retrieve (default: 10 for better coverage)
        
    Returns:
        List of dicts with 'content' and 'metadata'
    """
    # Initialize embeddings (must match the model used for storage)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Connect to existing Chroma database
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    # Create retriever with MMR (Maximal Marginal Relevance) search
    # MMR provides better diversity in results, reducing redundancy
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Use MMR instead of pure similarity
        search_kwargs={
            "k": k,  # Retrieve top 10 most relevant chunks
            "fetch_k": 30,  # Fetch 30 candidates before MMR filtering
            "lambda_mult": 0.7  # Balance between relevance (1.0) and diversity (0.0)
        }
    )
    
    # Retrieve documents
    docs = retriever.invoke(query)
    
    # Convert to simple dict format for state
    retrieved_docs = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]
    
    return retrieved_docs


# For testing the retriever independently
if __name__ == "__main__":
    print("Testing retriever node...")
    
    # Create initial state with a test query
    test_state: PipelineState = {
        "scraped_pages": None,
        "chunks": None,
        "stored_doc_ids": None,
        "query": "What are the research areas in the department?",
        "retrieved_docs": None,
        "response": None,
        "status": None,
        "error": None
    }
    
    # Run retriever
    result_state = retriever_node(test_state)
    
    # Print results
    if result_state.get("retrieved_docs"):
        print(f"\n✅ Retrieval successful!")
        print(f"Total documents: {len(result_state['retrieved_docs'])}")
        print(f"\nFirst document preview:")
        first_doc = result_state["retrieved_docs"][0]
        print(f"Source: {first_doc['metadata'].get('source', 'Unknown')}")
        print(f"Content (first 300 chars): {first_doc['content'][:300]}...")
    else:
        print(f"\n❌ Retrieval failed: {result_state.get('error')}")
