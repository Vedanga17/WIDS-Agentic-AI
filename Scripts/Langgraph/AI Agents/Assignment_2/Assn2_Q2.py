# making a simple LangGraph app with Groq LLM to analyze and answer user questions in two steps

# importing dependencies
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # for message types
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
    messages: List # to store all messages (human + AI) (will be useful when the output of the first agent is passed to the second one)

def analyzer(state: AgentState) -> AgentState:
    """ You will be given a question. Your job is to analyze it, and then rewrite it in a simpler manner.
    Make sure the rewritten question retains the original meaning but is easier to understand. You are NOT supposed to answer the question, just rewrite it.
    """
    
    # Create a system message with the analyzer instructions (docstring alone won't work, system message is the cleaner and better option)
    system_prompt = SystemMessage(content="""You will be given a question. Your job is to analyze it, and then rewrite it in a simpler manner.
Make sure the rewritten question retains the original meaning but is easier to understand. 
IMPORTANT: You are NOT supposed to answer the question, just rewrite it in simpler form. Only output the simplified question, nothing else.""")
    
    # Combine system message with the user's question (only the original question, not previous AI responses) and pass to the LLM.
    messages_for_analyzer = [system_prompt, state["messages"][0]] # VERY IMPORTANT: only the first message (user question) is passed to the analyzer.
    
    response = llm.invoke(messages_for_analyzer) # invoking the LLM with the "messages_for_analyzer" (system + user question). 
    print(f"\nAnalyzer response: {response.content}") # printing the simplified/rewritten question
    
    state["messages"].append(AIMessage(content=response.content)) # Append AI response to messages, so that the next agent can use the simplified question for answering.
    return state # returning the state

def generator(state:AgentState) -> AgentState:
    """ You are an expert in general knowledge. You will be given a simplified question by the 1st analyzer agent. 
    Your job is to provide a detailed and accurate answer to that question."""
    
    # Create a system message with the generator instructions (docstring alone won't work, system message is the cleaner and better option)
    system_prompt = SystemMessage(content="""You are an expert in general knowledge. You will be given a simplified question. 
Your job is to provide a detailed and accurate answer to that question.""")
    
    # Get the simplified question from the last AI message (from analyzer)
    simplified_question = state["messages"][-1].content
    
    # Create messages for generator: system instruction + simplified question
    messages_for_generator = [system_prompt, HumanMessage(content=simplified_question)] 
    # system prompt and the AI-simplified question is stored in this, to be passed on to the generator agent.
    
    result = llm.invoke(messages_for_generator) # invoking the LLM with "messages for generator" (system prompt + simplified question)
    print(f"\nGenerator response: {result.content}") # printing the answer to the simplified question

    return state 

graph = StateGraph(AgentState)

graph.add_node("Question Analyzer", analyzer)
graph.add_node("Answer Generator", generator)
               
graph.set_entry_point("Question Analyzer")
graph.add_edge("Question Analyzer", "Answer Generator")

graph.set_finish_point("Answer Generator")
               
agent = graph.compile()

user_input = input("Enter your question ('quit' to exit): ")

while user_input not in ["quit", "stop", "exit"]:

    message = HumanMessage(content=user_input) 
    state = {"messages": [message]} # adding user question to messages in state

    answer = agent.invoke({"messages": [message]}) # invoking the agent with the messages in state, including the human question. 

    user_input = input("\nEnter your question ('quit' to exit): ")