"""
Sequential Agent example.
This is an agentic AI architecture comprising of a sequential agent, having access to 3 sub agents to help the user decide
which car to buy, based on their company preference and their budget.
"""


from google.adk.agents import SequentialAgent # importing the Sequential Agent

from .sub_agents.company_checker.agent import company_checker
from .sub_agents.budget_checker.agent import budget_checker
from .sub_agents.suggester.agent import suggester # importing the 3 subagents from their respective folders

root_agent = SequentialAgent(
    name="car_selector_agent",
    sub_agents=[company_checker, budget_checker, suggester], # passing the 3 subagents to be used by the sequential agent in a list
    description="An AI assistant which suggests suitable car models on the basis of the user's preference."  
)

# first the company_checker agent will run, then budget_checker, and lastly the suggester agent.

# model name not required here (if you give it, program fails).