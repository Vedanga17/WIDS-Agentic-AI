"""
Football Striker Stats Agent
This is a parallel agentic workflow which summarizes the key statistics of the user provided footballer, which are number of goals
scored and favourite opponent to score against, best season (by goals), and number of titles won in career.

It is different from the sequential agent, as all the 3 subagents work together, and their results are combined by the 4th 
subagent to prepare a small report of the career.
"""

from google.adk.agents import ParallelAgent, SequentialAgent # importing Parallel and sequential agents

from .sub_agents.goals_agent.agent import goals_agent
from .sub_agents.season_agent.agent import season_agent
from .sub_agents.titles_agent.agent import title_agent
from .sub_agents.summarizer_agent.agent import summarizer_agent # importing subagents from their respective folders

footballer_agent = ParallelAgent(
    name="FootballerStatsAgent",
    sub_agents=[goals_agent, season_agent, title_agent], # the 3 parallel subagents passed as a list.
    description="""This agent uses its subagents to gather key information about the user given footballer and passes it to the
    summarizer agent to generate a comprehensive report about the footballer's career."""
)

root_agent = SequentialAgent(
    name="OverallReportAgent",
    sub_agents=[footballer_agent, summarizer_agent], # sequential agent containing the overall parallel agent workflow and the 
                                                     # summarizer agent
    description="""
    This agent combines the work of the parallel agent and the summarizer agent and gives the desired output.
    """
)

# model name not required here (if you give it, program fails).