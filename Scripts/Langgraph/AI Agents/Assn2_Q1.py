# Creating a simple LangGraph app with Groq LLM to answer user queries

# importing the dependencies
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage # for message types
from langchain_groq import ChatGroq # LLM model
from dotenv import load_dotenv
import os

load_dotenv() # for storing API key

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

class AgentState(TypedDict):
    messages: List # to store all messages (human + AI)

def process(state: AgentState) -> AgentState: # defining the agent process

    """ You are an expert in maths, coding and general knowledge. Answer the user queries to the best of your ability."""

    response = llm.invoke(state["messages"]) # invoking the LLM with the messages passed through state.
    print(f"\nAgent response: {response.content}")
    
    state["messages"].append(AIMessage(content=response.content)) # Append AI response to messages (for context)
    return state

graph = StateGraph(AgentState)

graph.add_node("LLM", process)
graph.set_entry_point("LLM")
graph.set_finish_point("LLM")

agent = graph.compile()

# Initialize message history (for context)
history = []

user_input = input("Enter your message ('quit' to exit): ")
while user_input not in ["quit", "stop", "exit"]:
    # Append user message to history
    history.append(HumanMessage(content=user_input))
    
    # Invoke agent with all messages (human + AI) to get its response
    result = agent.invoke({"messages": history}) # passed 'messages' through state, which also contains the history of the chat.
    
    # Update history with the result (includes AI response)
    history = result["messages"]
    
    user_input = input("\nEnter your message ('quit' to exit): ")