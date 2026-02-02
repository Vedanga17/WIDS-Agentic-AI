from google.adk.agents import Agent
from google.adk.tools import google_search # built-in ADK tool

DOB_giver = Agent(
    name="DOB_agent",
    model="gemini-2.5-flash-lite",
    description="This agent gives the date of birth of famous personalities using web search.",
    instruction="""
    You are a helpful AI assistant which gives the date of birth of famous personalities asked by the user.
    You have the access to the google search tool, to conduct a web search and fine the relevant date for the personality.
    ACTUALLY RETURN THE DATE OF BIRTH (DO NOT SIT IDLE) in the following format:

    {"name": "name of the personality", "dob": their Date of Birth in DD/MM/YYYY format}

    NOTE: If a user gives another prompt after you are called, and it is unrelated to the date of birth of a given personality, you 
    shall pass the responsibility back to the Manager to decide who to delegate the user's request too. You should do this every 
    time, without fail.
    """,
    tools=[google_search],
)

# In the instruction section, it is clearly mentioned that if the prompt given is unrelated to DOB, directly pass on the entire
# responsibility over to the manager agent to decide further flow. Very clear instructions given to the subagent here.

# This subagent can't be directly passed as a subagent, it must be wrapped inside the AgentTool method for the program to work.