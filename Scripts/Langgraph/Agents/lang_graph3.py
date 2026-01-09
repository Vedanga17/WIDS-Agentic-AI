#Learning to handle multiple nodes and edges in LangGraph

from typing import TypedDict, List
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    age: str
    skills: List[str]
    result: str

def name_node(state: AgentState) -> AgentState:

    state["result"] = "Hello, " + state["name"] + ", welcome to the system!"
    return state

def age_node(state: AgentState) -> AgentState:
    state["result"] = state["result"] + " You are " + state["age"] + " years old!"
    return state

def skills_node(state: AgentState) -> AgentState:
    skills_formatted = ""
    for i in range(len(state["skills"])):
        skills_formatted += state["skills"][i] + ", "
        if i == len(state["skills"]) - 1:
            skills_formatted = skills_formatted[:-2]  # Remove trailing comma and space
            skills_formatted += "!"

    state["result"] = state["result"] + " Your skills are: " + skills_formatted
    return state

graph = StateGraph(AgentState)

graph.add_node("name_node", name_node)
graph.add_node("age_node", age_node)
graph.add_node("skills_node", skills_node)

graph.set_entry_point("name_node")

graph.add_edge("name_node", "age_node") 
graph.add_edge("age_node", "skills_node")

graph.set_finish_point("skills_node")

app = graph.compile()

result = app.invoke({"name": "Bob", "age": "30", "skills": ["Python", "Machine Learning", "Data Analysis"]})
print(result["result"])