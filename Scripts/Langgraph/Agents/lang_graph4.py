#Learning to create conditional nodes in LangGraph

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    number1: int
    number2: int
    operation1: str
    number3: int
    number4: int
    operation2: str
    final_number: int
    final_number2: int

def add_node(state: AgentState) -> AgentState: # addition node
    """ This node adds the first 2 numbers. """

    state["final_number"] = state["number1"] + state["number2"]
    return state

def subtract_node(state: AgentState) -> AgentState: # subtraction node
    """ This node subtracts the 2nd number from the 1st number. """

    state["final_number"] = state["number1"] - state["number2"]
    return state

def add_node2(state: AgentState) -> AgentState: # addition node 2
    """ This node adds the 3rd and 4th numbers to the current final_number. """

    state["final_number2"] = state["number3"] + state["number4"]
    return state  

def subtract_node2(state: AgentState) -> AgentState: # subtraction node 2
    """ This node subtracts the 4th number from the 3rd number. """
    state["final_number2"] = state["number3"] - state["number4"]
    return state

def decide_next_node(state: AgentState) -> str: # router node 1 (based on operation1)
    """ This node decides which operation to perform based on the operation1 value. """

    if state["operation1"] == "+":
        return "addition_operation"
    else:
        return "subtraction_operation"

def decide_next_node2(state: AgentState) -> str: # router node 2 (based on operation2)
    """ This node decides which operation to perform based on the operation2 value. """

    if state["operation2"] == "+":
        return "addition_2"
    else:
        return "subtraction_2"

graph = StateGraph(AgentState)

graph.add_node("add_node", add_node)
graph.add_node("subtract_node", subtract_node)
graph.add_node("add_node2", add_node2)
graph.add_node("subtract_node2", subtract_node2)

graph.add_node("decide_next_node", lambda state:state)
graph.add_node("decide_next_node2", lambda state:state) 
# since this is a router node, we cannot pass an action to it, we pass a lambda function that returns the state as is.

graph.add_edge(START, "decide_next_node")

graph.add_conditional_edges( # adding conditional edges 
    "decide_next_node", # source node
    decide_next_node, # what action the node will perform
    {
        "addition_operation": "add_node", # edge: destination node
        "subtraction_operation": "subtract_node"
    }
)

graph.add_edge("add_node", "decide_next_node2")
graph.add_edge("subtract_node", "decide_next_node2") # adding edges to connect

graph.add_conditional_edges(
    "decide_next_node2",
    decide_next_node2,
    {
        "addition_2": "add_node2",
        "subtraction_2": "subtract_node2"
    }  
)

graph.add_edge("add_node2", END)
graph.add_edge("subtract_node2", END)

app = graph.compile()

lst = eval(input("Enter the 4 numbers as input: "))

initial_state = AgentState(
    number1=lst[0],
    number2=lst[1],
    operation1="-",
    number3=lst[2],
    number4=lst[3],
    operation2="+",
    final_number=0,
    final_number2=0
)

result = app.invoke(initial_state)

print("Final Result after first operation:", result["final_number"])
print("\nFinal Result after second operation:", result["final_number2"])