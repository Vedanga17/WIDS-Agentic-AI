"""
This is the root agent file, "manager agent" which decides which subagent will handle the user given prompt based on the 
work ability of the subagent. It controls the flow of information in the workflow. If the subagent gets confused, it just gives
the responsibility back to the manager for further execution and control.
"""

from google.adk.agents import Agent
import yfinance as yf # stock price tool
from google.adk.tools.agent_tool import AgentTool 
# Important: if a subagent uses a built-in ADK tool (for eg. google search), it cannot be directly passed as a subagent, it must 
# be wrapped inside the the AgentTool method, and then passed to the manager agent as a tool which it can use (ADK nonsense).

from .sub_agents.basic_math.agent import basic_math # importing the subagent from the folder it is present in
from .sub_agents.DOB_giver.agent import DOB_giver # importing the subagent from the folder it is present in

def current_stock_price(stock_name: str)-> dict: # custom tool for stock price display
    """This function is used to display the current stock price for a user-specified stock."""

    stock = yf.Ticker(stock_name)
    hist = stock.history(period="1d")
    latest_price = hist['Close'].iloc[-1]

    return {
        "Stock Price": f"the stock price of {stock_name} is ${latest_price:.2f}"
    }

root_agent = Agent(
    name="manager_agent",
    model="gemini-2.5-flash-lite",
    description=
    """This is a manager agent, which delegates work to multiple sub-agents on the basis of the prompt given to it by the user. """,
    instruction=
    """You are a helpful AI assistant who delegates work to sub-agents and uses tools to answer user questions.
    
    Available sub-agents:
    1. basic_math: Solves arithmetic problems (addition, subtraction, multiplication, division)
    2. DOB_giver: Finds date of birth of famous personalities using web search
    
    Available tools:
    - current_stock_price: Gets current stock price for a given stock symbol
    
    CRITICAL INSTRUCTIONS:
    1. When a user asks a question, identify which sub-agent or tool to use
    2. Delegate to the appropriate sub-agent OR use the appropriate tool
    3. ALWAYS display the result you receive to the user in your response
    4. DO NOT stay silent after receiving results from sub-agents or tools
    5. Format the answer in a clear, user-friendly way
    
    For example:
    - If user asks "what is 5+3", delegate to basic_math and then say "The answer is 8"
    - If user asks for stock price, use the tool and then say "The current stock price of AAPL is $150.25"
    - If user asks for DOB, delegate to DOB_giver and then say "The date of birth of [name] is [date]"
    
    You must ALWAYS provide a final response to the user with the information obtained.""",
    sub_agents=[basic_math],
    tools=[current_stock_price, AgentTool(DOB_giver)],
)

# subagents are passed in the sub_agents parameter.
# Instructions must be very clear and precise, so that the manager agent properly understands the responsibility bestowed upon it;
# it is given information about all the tools and subagents at its disposal.

