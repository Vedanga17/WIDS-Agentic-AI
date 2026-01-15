# creating a simple LangGraph app with an LLM to redirect user questions to expert agents, based on the domain of the question

# importing the dependences

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage # for message types
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
    messages: List[HumanMessage] # to store human messages

def python_expert(state: AgentState) -> AgentState:
    """ You are an expert Python programmer. Answer the user's Python-related questions accurately and concisely."""

    # Create a system message with the python expert instructions
    system_prompt = SystemMessage(content="""You are an expert Python programmer. Answer the user's Python-related questions accurately and concisely.""")
    
    state["messages"].append(SystemMessage(content=system_prompt.content))  # Append system prompt to messages for context

    response = llm.invoke(state["messages"]) # invoking the LLM with the messages passed through state.
    print(f"\nPython Expert response: {response.content}")
    
    return state

def general_expert(state: AgentState) -> AgentState:
    """ You are an expert in general knowledge. Answer the user's general knowledge questions accurately and concisely."""

    # Create a system message with the general expert instructions
    system_prompt = SystemMessage(content="""You are an expert in general knowledge. Answer the user's general knowledge questions accurately and concisely.""")
    
    state["messages"].append(SystemMessage(content=system_prompt.content))  # Append system prompt to messages for context

    result = llm.invoke(state["messages"]) # invoking the LLM with the messages passed through state.
    print(f"\nGeneral Expert response: {result.content}")
    
    return state

def decide_expert(state: AgentState) -> str:
    """ Use LLM to intelligently decide whether the question is related to Python programming or general knowledge."""

    question = state["messages"][-1].content  # Get the latest user question

    # Create a classification prompt for the LLM
    decider_prompt = SystemMessage(content="""You are a question classifier. Analyze the user's question and determine if it's about:
- Python programming (code, syntax, libraries, debugging, algorithms in Python context, etc.) -> respond with only the word "python"
- General knowledge (history, science, math, facts, non-programming topics, etc.) -> respond with only the word "general"

Output ONLY one word: either "python" or "general". Nothing else.""")
    
    classification_prompt = [decider_prompt, HumanMessage(content=question)] # combining system prompt and user question and pass to the LLM for classfication.
    
    
    # Ask the LLM to classify (even if I use python keywords in a general sense, it will be able to classify correctly)

    classification = llm.invoke(classification_prompt) # passed the prompt to the LLM; will return either "python" or "general"
    decision = classification.content.strip().lower() # returns a list with either "python" or "general"
    
    print(f"\nRouter decision: {decision}")
    
    # Return the classification
    if "python" in decision:
        return "python"
    else:
        return "general"
    
graph = StateGraph(AgentState)

graph.add_node("Router agent", lambda state:state) # first node present, decides where to direct the user query
graph.add_node("Python Expert", python_expert)
graph.add_node("General Expert", general_expert)

graph.add_edge(START, "Router agent")

graph.add_conditional_edges(
    "Router agent", # source node
    decide_expert, # condition to be applied to determine way forward
    {
        "python": "Python Expert", # edge: node
        "general": "General Expert" 
    }
)

graph.add_edge("Python Expert", END)
graph.add_edge("General Expert", END)

agent = graph.compile()

user_input = input("Enter your question ('quit' to exit): ")

while user_input not in ["quit", "exit", "stop"]:

    question = HumanMessage(content=user_input) # storing the user question in a variable which will be passed to the state, and then the agent.
    state = {"messages": [question]} # adding user question to messages in state

    answer = agent.invoke({"messages": [question]}) # invoking the agent with the messages in state, including the human question.
    
    user_input = input("\nEnter your question ('quit' to exit): ")