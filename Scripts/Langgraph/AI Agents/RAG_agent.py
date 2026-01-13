# Creating a RAG agent that can use a vector database to answer questions

# importing the dependencies

from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages # reducer function
from langchain_groq import ChatGroq # LLM model
from langchain_huggingface import HuggingFaceEmbeddings # embedding model
from langchain_community.document_loaders import PyPDFLoader # for loading the pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter # for splitting the text into chunks
from langchain_chroma import Chroma # for vector store
from langchain_core.tools import tool

load_dotenv()  # for storing API key

llm_ = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0, 
    api_key=os.getenv("GROQ_API_KEY")) # setting temp = 0 to minimize hallucinations.

# our embedding model - and it is compatible with our LLM as well
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build path relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "SkillX_Quant_Finance_Session1.pdf") # our vector store document.

pdf = PyPDFLoader(pdf_path) # loading the PDF document

# using a try and except block to handle any errors during page splitting
try:
    pages = pdf.load() # splitting the pdf into pages
    print(f"Successfully loaded {len(pages)} pages from the PDF document.")

except Exception as e:
    print(f"Error loading PDF document: {e}")
    raise

# Chunking process (splitting pages into smaller 'chunks' for better embedding)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

pages_split = text_splitter.split_documents(pages) # Now we apply this to the pages of our document (to break into chunks).


persist_directory = "chroma_rag_db" # location to store the vector database
collection_name = "skillx_session_info" # name of the collection in the vector DB


vector_store = Chroma.from_documents( # creating the vector store with relevant parameters
    documents=pages_split, # obtains the chunks of the document
    embedding=embeddings, # embeds the chunks
    persist_directory=persist_directory, # directory to persist the vector store
    collection_name=collection_name # name of the collection in the vector store
)
print("Successfully created the vector store!")

# now have to create our retriever 
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4} # setting number of chunks to retrieve
)

@tool
def retriever_tool(query: str) -> str: # creating the retriever tool to be used by the agent
    """ Tool to retrieve and return relevant information from the vector store based on the query. """
    docs = retriever.invoke(query) # retrieving the relevant info

    if not docs:
        return "No relevant information found in the document." # no similarities found
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Chunk {i+1}:\n{doc.page_content}\n") # if relevant info is found, return it in a formatted manner for the agent to use.
    
    return "\n\n".join(results) # joining the results

tools = [retriever_tool] # storing the tool in a list for the agent to use.

llm = llm_.bind_tools(tools) # binding the tool to the LLM

class AgentState(TypedDict): # making the state class.
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> str:
    """ Checks if the last message contains any tool calls or not. """
    last_message = state["messages"][-1]
    if not last_message.tool_calls:  # if no tool calls were made, end the process
        return "end"
    else:
        return "continue"  # otherwise, continue.
    
system_prompt = """
You are an intelligent AI assistant who answers questions about the SkillX Quant Finance Session 1 slides based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the SkillX Quant Finance Session 1 slides. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
PLEASE always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# now time to define the LLM agent.
def LLM_agent(state: AgentState) -> AgentState:
    messages = state["messages"] # getting the messages from the state
    system_message = SystemMessage(content=system_prompt) # adding the system prompt
    messages += [system_message] # getting the complete message by combining previous messages and system prompt
    response = llm.invoke(messages) # obtaining the LLM's response by using the invoke method
    return {"messages": [response]} # returning the response as the new state

# now time to create the retriever agent as well
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls # Getting the tool calls from the latest message in state.
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}") # storing tool execution results by iterating through "tool_calls"
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools." # tells us to use a tool which is actually present in the set of given tools
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', '')) # if tool is present, we invoke it with the user-given query.
            print(f"Result length: {len(str(result))}") # print the length of the result.
            

        # Appends the Tool Message, including tool call ID, name, and content
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results} # returns the state with the tool execution results.

graph = StateGraph(AgentState)

graph.add_node("LLM_agent", LLM_agent) # adding the LLM agent node
graph.set_entry_point("LLM_agent")

graph.add_node("retriever_agent", take_action) # adding the retriever agent node
graph.add_edge("retriever_agent", "LLM_agent") # adding edge from retriever to LLM agent

graph.add_conditional_edges( # adding conditional edges based on whether to continue or end
    "LLM_agent", # source node
    should_continue, # condition function
    {
        "continue": "retriever_agent", # edge to retriever agent if continuing
        "end": END  # edge to end if ending
    }
)

rag_agent = graph.compile()

def running_agent(): # function to run the RAG agent and accept multiple input queries
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question?: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages}) # invoking the RAG agent with the user input
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content) # printing only the relevant content of the final message.


running_agent()