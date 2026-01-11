#First introduction to AI Agents using Langgraph and Langchain with Groq LLM

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import time

load_dotenv() # for storing API key

# Initialize Groq with fast inference
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

class AgentState(TypedDict):
    messages: List[HumanMessage] # for making agentstate class

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\n Agent response: {response.content}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process_node", process)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)

agent = graph.compile()

user_input = input("Enter your message: ")
while user_input != "exit":
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    print(result["messages"][-1].content)
    time.sleep(2)  # Add delay to prevent token exhaustion
    user_input = input("Enter your message: ")