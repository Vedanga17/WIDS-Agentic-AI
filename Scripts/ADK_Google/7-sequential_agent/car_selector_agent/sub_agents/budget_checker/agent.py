"""
Budget Checker Agent.
This agent checks what is the budget range specified by the user, so that an appropriate car can be suggested by the next agent.
It stores the budget in state to be passed forward to the next agent.
"""

from google.adk.agents import LlmAgent

budget_checker = LlmAgent(
    name="BudgetCheckerAgent",
    model="gemini-2.5-flash-lite",
    description="This is an AI agent which checks the budget specified by the user for the car purchase and stores in state.",
    instruction="""
    You are a budget checker agent. You will check the budget specified by the user for their car purchase, and store the numbers
    in state.
    OUTPUT ONLY a list containing the upper and lower bounds of the user's budget (budget is in Indian rupees).
    Example: [2000000, 4000000], or [10000000, 15000000]
    """,
    output_key="budget", # storing the result of the agent to state, for the next subagent to access.
)