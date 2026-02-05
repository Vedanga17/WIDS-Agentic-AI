"""

Departmental Knowledge Assistant Agent, combining 5 agents: 
Scraper Agent: Scrapes the department website and stores data in a raw format.
Processor Agent: Processes (chunks and embeds) the data, and converts it into a legible format, and stores it in a 
vector database (Chroma Vector store).
Retriever Agent: Gets the user query and retrieves relevant information from the vector store
to use for responding.
Responder Agent: Summarizes the data obtained from the retriever to draft a suitable response
to give to the user.
Orchestrator Agent: Orchestrates the entire workflow via LangGraph.

Streamlit Frontend for easier accessibility for the user.
"""

from langgraph.graph import StateGraph, START, END

from state import PipelineState
from nodes.processor_node import processor_node
from nodes.scraper_node import scraper_node
from nodes.retriever_node import retriever_node
from nodes.responder_node import responder_node

# ===== DATA COLLECTION WORKFLOW =====
# Run this ONCE to scrape and store data
def build_data_collection_graph():
    """Build graph for scraping and storing data (one-time setup)"""
    graph = StateGraph(PipelineState)
    
    graph.add_node("scraper", scraper_node)
    graph.add_node("processor", processor_node)
    
    graph.add_edge(START, "scraper")
    graph.add_edge("scraper", "processor")
    graph.add_edge("processor", END)
    
    return graph.compile()

# ===== QUERY WORKFLOW =====
# Run this for each user query
def build_query_graph():
    """Build graph for answering queries (used repeatedly)"""
    graph = StateGraph(PipelineState)
    
    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", responder_node)
    
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "responder")
    graph.add_edge("responder", END)
    
    return graph.compile()

# Create both apps
data_collection_app = build_data_collection_graph()
query_app = build_query_graph()

# ===== FUNCTIONS FOR APP.PY =====

def collect_data():
    """
    Run data collection workflow (scraping + processing)
    Call this ONCE to initialize the database
    """
    print("Starting data collection...")
    initial_state: PipelineState = {
        "scraped_pages": None,
        "pdf_files": None,
        "chunks": None,
        "stored_doc_ids": None,
        "query": None,
        "retrieved_docs": None,
        "response": None,
        "status": None,
        "error": None
    }
    result = data_collection_app.invoke(initial_state)
    return result


def run_query(query: str) -> str:
    """
    Run query workflow (retrieval + response generation)
    Call this for each user question
    
    Args:
        query: User's question
        
    Returns:
        Generated response
    """
    initial_state: PipelineState = {
        "scraped_pages": None,
        "pdf_files": None,
        "chunks": None,
        "stored_doc_ids": None,
        "query": query,
        "retrieved_docs": None,
        "response": None,
        "status": None,
        "error": None
    }
    result = query_app.invoke(initial_state)
    return result["response"]


# For testing
if __name__ == "__main__":
    # Step 1: Collect data (run once)
    # Uncomment below to scrape and process data
    # print("=== COLLECTING DATA ===")
    # collect_data()
    
    # Step 2: Test query (run multiple times)
    print("\n=== TESTING QUERY ===")
    response = run_query("What are the research areas in the department?")
    print(f"\nResponse: {response}")



