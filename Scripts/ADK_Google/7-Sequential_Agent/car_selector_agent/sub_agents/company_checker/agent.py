"""
Company Checker Agent
This is an agent which checks which company the user likes, and wants his car to be from.

"""

from google.adk.agents import LlmAgent

company_checker=LlmAgent(
    name="CompanyCheckerAgent",
    model="gemini-2.5-flash-lite",
    description="This agent checks the user's preferred car company (the one which they buy from) and store it in state.",
    instruction="""
    You are a Company Checker Agent. You will check which company the user likes, and wants to buy a car from.
    OUTPUT ONLY the name of the COMPANY.
    Example: 'Mercedes', or 'Audi' etc.
    """,
    output_key="company",
)