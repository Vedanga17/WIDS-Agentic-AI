# Creating a simple Drafter Agent using Groq API and Langgraph

# importing the dependencies

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages # reducer function
from langchain_groq import ChatGroq 
from langgraph.prebuilt import ToolNode # node which stores tools used by the agent
from dotenv import load_dotenv
import os

load_dotenv() # for storing API key

# We require a global variable to hold the document content being drafted. 
document_content = ""


class AgentState(TypedDict): # making the state class.
    messages: Annotated[Sequence[BaseMessage], add_messages]

# defining the 2 tools to be used: update and save.
@tool
def update(content: str) -> str:
    """Updates the document by replacing it with the user-given content."""
    global document_content # specifying that we are using the global variable
    document_content = content
    return f"Document has been updated. The current content is: \n{document_content}"

@tool
def save(filename: str) -> str:
    """ Saves the current document to a textfile and then end the task.
    
    Argument
    filename: the name of the textfile where the content will be stored and saved."""

    global document_content

    if not filename.endswith('.txt'): # ensure the file has a .txt extension
        filename += '.txt'
    with open(filename, 'w') as file:
        file.write(document_content)
    print("\n Document has been saved to the file: ", filename)
    return f"Document has been saved to the file: {filename}. Task is now complete." # task completion message.

tools = [update, save] # stored the tools in a list for the agent to use.

# Initialize Groq LLM
llm_ = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Bind tools to the LLM
llm = llm_.bind_tools(tools)

def agent(state: AgentState) -> AgentState: # defining the agent and its tasks. will be used for the node.
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and FINISH, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        # First interaction - just greet the user
        print("\nHello! I'm Drafter, your assistant for drafting documents. How can I help you today?")

        user_input = input("\nWhat would you like to do with the document? ") # asking what the user wants to do
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document?") # the document is already made, and it will be modified as per instructions.
        user_message = HumanMessage(content=user_input)
     
    cumulative_message = [system_prompt] + list(state["messages"]) + [user_message] # the entire message history, which will be passed to the agent.
    response = llm.invoke(cumulative_message)

    if response.content:  # Only print if there's actual content
        print(f"\nAgent response: {response.content}") # printing the agent's response.

    return {"messages": list(state["messages"]) + [user_message, response]} # returning the updated messages (updating the state). 
    
def should_continue(state: AgentState) -> str: # conditional settings for the agent.
    """ Determine if the agent should continue or end based on tool calls. """ 

    messages = state["messages"]

    if not messages:
        return "continue" # Since the state is empty, we have to start the process; nothing to check here.
    
    # Now we have to see which is the last message in the state: update or save, and then do the action accordingly.
    
    for message in reversed(messages): # most recent tool usage being checked here.
        if isinstance(message, ToolMessage) and \
            "saved" in message.content.lower() and \
            "document" in message.content.lower():
            return "end" # if the last tool used was 'save', we end the process.
        else:
            return "continue" # if not, we continue further (not the end yet)
    
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages: # if there are no messages, there is nothing to print; just return.
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content} \n") # if there is a tool message, print it.


graph = StateGraph(AgentState)
graph.add_node("drafter_agent", agent)
graph.set_entry_point("drafter_agent")

graph.add_node("tools", ToolNode(tools=tools))

# adding conditional edges between the drafter agent and the tools node, and ending when the "save" tool is used.
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "drafter_agent",
        "end": END
    }
)

graph.add_edge("drafter_agent", "tools")

app = graph.compile()

def run_doc_agent(): # function to run the drafter agent.
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_doc_agent()


