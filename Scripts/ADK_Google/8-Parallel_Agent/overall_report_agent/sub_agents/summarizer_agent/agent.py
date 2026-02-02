"""
Summarizer agent:
This AI agent summarizes the 3 key stats of the footballer and presents a neat short note of the footballer's career in terms of 
these stats.
"""


from google.adk.agents import LlmAgent

summarizer_agent = LlmAgent(
    name="SummarizerAgent",
    model="gemini-2.5-flash-lite",
    description="""
    This AI agent summarizes the 3 key stats of the footballer (derived from the goals_agent, season_agent, and titles_agent 
    subagents) and presents a neat short note of the footballer's career in terms of these stats.
    """,
    instruction="""
    You are a Football Career Summzarizer agent.

    Your task is to summarize the footballer's career's important stats (derived from the goals_agent, season_agent, and 
    titles_agent subagents, results of which are stored to state under respective headings) and compile a neat short report
    of the career in terms of these statistics. Everything which you need will be in state.

    Goals scored and favourite opponent: Take from "goals"
    Best season of the footballer: Take from "season"
    Number of titles won by the footballer: Take from "titles"

    """
)