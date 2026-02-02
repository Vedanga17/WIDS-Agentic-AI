"""
Season Agent:
This agent finds the best season of the footballer, in terms of goals and assists scored.
"""

from google.adk.agents import LlmAgent

season_agent = LlmAgent(
    name="SeasonAgent",
    model="gemini-2.5-flash-lite",
    description="""
    This AI agent finds the best season of the footballer, in terms of goals and assists scored.
    """,
    instruction="""
    You are a Football Season agent.

    Your task is to look up the best season of the user given footballer (here season means calendar year, not the regular 
    football season) in terms of the number of GOALS AND ASSISTS scored in that year. Return the calendar year and the number of 
    GOALS AND ASSISTS scored.
    Store the result to state, under the header given in the output_key parameter.

    Example: Lionel Messi's best season was 20XX, in which he scored Y goals and Z assists, for a total of W goals and assists.
    """,
    output_key="season", # saving the result to state for the summarizer agent to access.
)