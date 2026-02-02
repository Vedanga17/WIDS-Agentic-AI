"""
Titles Agent:
This agent finds the number of titles won by the footballer throughout their career (international + club).
"""


from google.adk.agents import LlmAgent

title_agent = LlmAgent(
    name="TitlesAgent",
    model="gemini-2.5-flash-lite",
    description="""
    This AI agent finds the number of titles won by the footballer throughout their career (international + club).
    """,
    instruction="""
    You are a Football Titles agent.

    Your task is to look up the number of titles won by the user given footballer throughout their career, and segregate the 
    numbers into international titles and club titles. Return the number of titles won in both the given formats. Store the 
    result to state, under the header given in the output_key parameter.

    Example: Lionel Messi has won X titles to date: Y international titles and Z club titles.
    """,
    output_key="titles", # saving the result to state for the summarizer agent to access.
)