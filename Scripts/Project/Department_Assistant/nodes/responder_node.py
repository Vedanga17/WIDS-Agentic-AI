"""
Responder Node - Answer Generation Component
This node generates final responses using LLM based on retrieved documents
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS
from state import PipelineState

# Load environment variables
load_dotenv()


def responder_node(state: PipelineState) -> PipelineState:
    """
    Main responder node function
    Generates answer using LLM based on query and retrieved documents
    
    Args:
        state: Current pipeline state with query and retrieved_docs
        
    Returns:
        Updated state with response
    """
    print("💬 Generating response...")
    
    try:
        # Get query and retrieved docs from state
        query = state.get("query")
        retrieved_docs = state.get("retrieved_docs", [])
        
        if not query:
            raise ValueError("No query found in state")
        
        if not retrieved_docs:
            print("⚠️  No documents retrieved, generating response without context")
        
        # Generate response
        response = generate_response(query, retrieved_docs)
        
        print(f"✅ Response generated ({len(response)} characters)")
        
        # Update state
        state["response"] = response
        state["status"] = "Successfully generated response"
        
    except Exception as e:
        state["error"] = f"Response generation failed: {str(e)}"
        state["status"] = "Failed"
        print(f"❌ Response generation error: {str(e)}")
    
    return state


def generate_response(query: str, retrieved_docs: list) -> str:
    """
    Generate answer using LLM with retrieved context
    
    Process:
    1. Format retrieved documents as context
    2. Create system prompt with instructions
    3. Create user message with query and context
    4. Get LLM response
    
    Args:
        query: User's question
        retrieved_docs: List of relevant documents from vector DB
        
    Returns:
        Generated answer as string
    """
    # Initialize Groq LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
        max_tokens=MAX_TOKENS
    )
    
    # Format retrieved documents as context
    context = format_context(retrieved_docs)
    
    # Create system prompt
    system_prompt = """You are an intelligent AI assistant for the IIT Bombay Chemical Engineering Department.
Your role is to answer questions about the department based on information from the department website.

Instructions:
- Answer questions accurately using the provided context
- Carefully read through ALL provided context documents - the answer may be spread across multiple documents
- If you find relevant information (even in just one document), provide a complete answer
- If the context doesn't contain relevant information, politely say you don't have that information
- For questions about counts or numbers, look carefully for numerical information in the context
- Synthesize information from multiple documents when needed
- Always cite the sources (URLs) from the context when possible
- Be concise but informative
- Use a helpful and professional tone
"""
    
    # Create user message with query and context
    user_message = f"""Question: {query}

Context from department website:
{context}

Please provide a comprehensive answer based on the context above."""
    
    # Create messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    # Get LLM response
    response = llm.invoke(messages)
    
    return response.content


def format_context(retrieved_docs: list) -> str:
    """
    Format retrieved documents into a readable context string
    
    Args:
        retrieved_docs: List of dicts with 'content' and 'metadata'
        
    Returns:
        Formatted context string
    """
    if not retrieved_docs:
        return "No relevant information found in the database."
    
    formatted_parts = []
    
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get('metadata', {}).get('source', 'Unknown source')
        content = doc.get('content', '')
        
        formatted_parts.append(f"[Document {i}]\nSource: {source}\nContent: {content}\n")
    
    return "\n".join(formatted_parts)


# For testing the responder independently
if __name__ == "__main__":
    print("Testing responder node...")
    
    # Create sample retrieved documents
    test_retrieved_docs = [
        {
            "content": "The Chemical Engineering department offers B.Tech, M.Tech, and PhD programs. Research areas include reaction engineering, process systems, and biological systems engineering.",
            "metadata": {"source": "https://www.che.iitb.ac.in/programs"}
        },
        {
            "content": "Faculty members specialize in various areas such as fluid mechanics, thermodynamics, soft matter engineering, and catalysis.",
            "metadata": {"source": "https://www.che.iitb.ac.in/faculty"}
        }
    ]
    
    # Create initial state
    test_state: PipelineState = {
        "scraped_pages": None,
        "chunks": None,
        "stored_doc_ids": None,
        "query": "What programs does the department offer?",
        "retrieved_docs": test_retrieved_docs,
        "response": None,
        "status": None,
        "error": None
    }
    
    # Run responder
    result_state = responder_node(test_state)
    
    # Print results
    if result_state.get("response"):
        print(f"\n✅ Response generation successful!")
        print(f"\nQuery: {test_state['query']}")
        print(f"\nResponse:\n{result_state['response']}")
    else:
        print(f"\n❌ Response generation failed: {result_state.get('error')}")
