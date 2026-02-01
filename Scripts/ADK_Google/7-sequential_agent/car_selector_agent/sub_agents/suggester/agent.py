"""
Suggester Agent.
On the basis of the company name and the budget (obtained from state), this agent suggest a suitable car which the user can buy. 
"""

from google.adk.agents import LlmAgent

suggester = LlmAgent(
    name="SuggesterAgent",
    model="gemini-2.5-flash-lite",
    description="""This is an AI agent which suggest a suitable car which the user can buy based on their company preference and
    budget range (company and budget obtained from state).""",
    instruction="""
    You are a car suggester agent. Based on the company preference and the budget of the user, suggest a suitable car name which
    the user can purchase. The budget range and the company preference is present in state and will be automatically available to you.

    OUTPUT ONLY the name of the car, and the car variant.
    Example: A suitable car for you is Honda City V.
    OR 
    A suitable car for you is BMW X5 SUV.
    etc.
    """,
    output_key="suggestion",
)