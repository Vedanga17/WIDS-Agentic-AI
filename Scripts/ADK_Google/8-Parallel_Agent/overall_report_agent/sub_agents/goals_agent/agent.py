"""
Goals Agent:
This agent finds the total number of goals scored by the footballer, and the favourite opponent by number of goals scored.
"""

from google.adk.agents import LlmAgent

goals_agent = LlmAgent(
    name="GoalsAgent",
    model="gemini-2.5-flash-lite",
    description="""
    This AI agent finds the total number of goals scored by the footballer, and the favourite opponent by number of goals scored.
    """,
    instruction="""
    You are a Football Goals agent.

    Your task is to look up the total number of goals scored by the footballer in their career, and also to find their favourite
    opponent by goals scored, i.e. the opponent team against which they have scored the highest number of goals.
    Return the total number of goals scored in career, as well as their favourite opponent and the number of goals scored against
    them. Store the result to state, under the header given in the output_key parameter.

    Example: Lionel Messi has scored X goals in his career. His favourite opponent is Real Madrid, scoring Y goals against them.
    """,
    output_key="goals", # saving the result to state for the summarizer agent to access.
)