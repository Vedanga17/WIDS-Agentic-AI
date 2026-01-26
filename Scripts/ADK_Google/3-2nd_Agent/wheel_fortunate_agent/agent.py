import random # to randomly pick which good outcome to display
from google.adk.agents import Agent

def fortunate_wheel(): # display 1 of the 3 good outcomes
    outcomes = [
        "You win coupons for a free trip to the Bahamas!",
        "You win $5000!",
        "You get to meet your favourite celebrity and hang out with them!"
    ]
    return random.choice(outcomes)

root_agent = Agent(
    name= "wheel_fortunate_agent",
    model= "gemini-2.5-flash-lite",
    description= "Agent which spins a wheel and chooses and displays a good outcome.",
    instruction= """

    You are an AI assistant, and you have to use the tool provided to display the good outcome.
    After choosing the tool, you have to DISPLAY it as well, in the output.
    CRITICAL: You MUST reply to the user with the result you received from the tool.
    DO NOT STAY SILENT, actually DISPLAY the outcome. 

    Provided tools: fortunate_wheel""", # giving the agent very specific instructions
    tools=[fortunate_wheel] # passing the list of available tools 
)