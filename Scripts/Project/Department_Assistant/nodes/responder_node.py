"""
Responder Node - Answer Generation Component
This node generates final responses using LLM based on retrieved documents
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS, CHAT_HISTORY_TURNS
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
        # Get query, retrieved docs, and prior conversation turns from state
        query = state.get("query")
        retrieved_docs = state.get("retrieved_docs", [])
        chat_history = state.get("chat_history") or []

        if not query:
            raise ValueError("No query found in state")

        if not retrieved_docs:
            print("⚠️  No documents retrieved, generating response without context")

        # Generate response
        response = generate_response(query, retrieved_docs, chat_history)

        print(f"✅ Response generated ({len(response)} characters)")

        # Update state
        state["response"] = response
        state["status"] = "Successfully generated response"

    except Exception as e:
        state["error"] = f"Response generation failed: {str(e)}"
        state["status"] = "Failed"
        print(f"❌ Response generation error: {str(e)}")

    return state


def generate_response(query: str, retrieved_docs: list, chat_history: list = None) -> str:
    """
    Generate answer using LLM with retrieved context

    Process:
    1. Format retrieved documents as context
    2. Create system prompt with instructions
    3. Convert recent chat history into prior turns, so follow-up questions can
       be understood in context
    4. Create user message with query and context
    5. Get LLM response

    Args:
        query: User's question
        retrieved_docs: List of relevant documents from vector DB
        chat_history: Prior turns of the conversation, oldest first, as
            [{"role": "user"/"assistant", "content": "..."}, ...]. Only the most
            recent CHAT_HISTORY_TURNS exchanges are used - this is meant to let
            the model follow the thread of a conversation (resolve "the other
            one", "what about X instead", etc.), not to grow the prompt without
            bound.

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
- Be concise but informative
- Use a helpful and professional tone
- Write the answer itself as plain, natural prose - do not interrupt sentences with inline
  citations, parenthetical source references, or bracketed markers of any kind (for example,
  do NOT write things like "(Source: https://...)" or "[1]" in the middle of a sentence)

Using conversation history:
- If earlier turns of this conversation are included below, use them only to understand what
  the current question is referring to (e.g. "the other one", "what about that program?", "and
  faculty?") - resolve the reference, then answer using the provided context documents
- Every factual claim in your answer must still come from the context documents given for THIS
  question, not from what was said earlier in the conversation - a prior answer is not itself a
  source, so do not restate or lean on facts from earlier turns unless the current context also
  supports them

Citing sources:
- If, and only if, the retrieved context actually contains a "Source:" URL for information you
  used, end your answer with a short "Sources:" section listing those exact URLs as plain text,
  one per line
- Do NOT invent your own citation notation, reference IDs, footnote markers, or line-number
  annotations (for example, bracketed tags like [1] with a made-up numbering scheme, or any
  similar symbols) - nothing in the provided context has line numbers or reference IDs, so any
  such marker you produce is fabricated and misleading
- If no context was provided or none of it was actually used in the answer, omit the "Sources:"
  section entirely rather than including an empty or speculative one
"""

    # Create user message with query and context
    user_message = f"""Question: {query}

Context from department website:
{context}

Please provide a comprehensive answer based on the context above."""

    # Convert the most recent chat turns into actual conversation messages, so the
    # model sees them as prior dialogue rather than a wall of text. Capped to the
    # last CHAT_HISTORY_TURNS exchanges (2 messages per exchange: user + assistant).
    history_messages = []
    if chat_history:
        for turn in chat_history[-(CHAT_HISTORY_TURNS * 2):]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                history_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                history_messages.append(AIMessage(content=content))

    # Create messages: system instructions, then prior conversation turns (if
    # any), then the current question with its retrieved context
    messages = [SystemMessage(content=system_prompt)] + history_messages + [
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
        "chat_history": [],
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
