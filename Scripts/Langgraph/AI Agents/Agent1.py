#First introduction to AI Agents using Langgraph and Langchain with Groq LLM

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv() # for storing API key

# Initialize ChatGroq with your API key
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
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
    user_input = input("Enter your message: ")