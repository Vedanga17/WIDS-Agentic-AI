from datetime import datetime
from google.adk.agents import Agent # agent
from google.adk.tools import google_search # built-in tool

def get_current_time() -> dict: # custom tool
    """ Get the current time and return it in the form of a dictionary."""
    return {
        "current time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def factorial(a) -> dict: # custom tool
    """Given a number, return its factorial."""
    fact = 1
    if a == 0 or a==1:
        return fact
    else:
        for i in range(1, a+1):
            fact *= i
        return fact
    
# defining our root_agent
root_agent = Agent(
    name = "tool_agent",
    model="gemini-2.5-flash",
    description="Agent which uses given tools",

    instruction="""
    You are a helpful AI assistant. Use the provided tools to return results to user queries.
    Provided tools= factorial AND get_current_time""",

    tools=[factorial, get_current_time] # we can pass in 2 custom built tools

    # tools=[factorial, google_search] # we cannot do this; ADK doesn't allow a built-in tool and custom tool to be passed together
)

