#Learning the simplest LangGraph structure

from typing import Dict, TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str

def complimenting_node(state: AgentState) -> AgentState: 
    """ Simple node that compliments the user. """

    state['name'] = state['name'] + ", you are doing a great job!"

    return state

graph = StateGraph(AgentState)
graph.add_node("complimenter", complimenting_node)

graph.set_entry_point("complimenter")
graph.set_finish_point("complimenter")

app = graph.compile()

result = app.invoke({"name": "Alice"})
print(result["name"])


