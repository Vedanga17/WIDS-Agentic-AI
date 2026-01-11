#Creating a simple ReAct agent using Groq LLM in Langgraph

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import time
import os

load_dotenv() # for storing API key

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a,b):
    """ Adds 2 numbers together."""
    return a+b

@tool
def subtract(a,b):
    """ Subtracts b from a."""
    return a-b

@tool
def multiply(a,b):
    """ Multiplies 2 numbers together."""
    return a*b

@tool
def exponentiate(a,b):
    """ Raises a to the power b."""
    return a**b

#defined the 4 tools to be used.

tools = [add, subtract, multiply, exponentiate]


# Initialize Groq LLM
llm_ = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Bind tools to the LLM
llm = llm_.bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are a helpful AI agent. When you use tools to perform calculations, ALWAYS include the calculation results in your final response to the user. Be complete and thorough in your answers.")
    response = llm.invoke([system_prompt] + state["messages"])   
    return {"messages": [response]}

# Function to determine whether to continue or end the agent's operation
def should_continue(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)

graph.add_node("agent", model_call)
graph.set_entry_point("agent")

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()

# function to print the output in a proper manner, including tool calls, ai messages, etc.
def print_stream(stream):
    for s in stream:
        message =  s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

query = input("Enter your question here: ")

# adding a while loop so that the user can ask as many questions as they want.
while query != "exit":
    inputs = {"messages": [("user", query)]}
    print_stream(app.stream(inputs, stream_mode="values"))
    time.sleep(2)  # slight delay before next input
    query = input("Enter your question here: ")
