from google.adk.agents import Agent

# defining our root agent
root_agent = Agent(
    name="greeting_agent", # the agent name MUST match the folder name, else an error will arise.
    model="gemini-2.5-flash",
    description="Greeting Agent",
    instruction="""
    You are a helpful assistant that greets the user.
    Ask for the user's name and greet them by name.
    """,
)
    
