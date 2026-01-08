from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    values: List[int]
    operation: str
    result: str

def multiply(lst):
    result = 1
    for number in lst:
        result *= number
    return result

def process_values(state: AgentState) -> AgentState: 
    """ Node that performs a set of calculations."""

    if state['operation'] == '+':
        state['result'] = f"Hi {state['name']}! Your answer is : {sum(state['values'])}"
    else:
        state['result'] = f"Hi {state['name']}! Your answer is : {multiply(state['values'])}"
    
    return state

graph = StateGraph(AgentState)

graph.add_node("processor", process_values)

graph.set_entry_point("processor")
graph.set_finish_point("processor")

app = graph.compile()

#from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png))      

result = app.invoke({"name": "Vedanga", "values": [1, 2, 3, 4], "operation": '+'})

print(result["result"])

