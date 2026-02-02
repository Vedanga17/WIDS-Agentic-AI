from google.adk.agents import Agent

def Add(a: int, b: int)-> dict: # defining the 4 arithmetic tools (custom tools)
    """Add 2 integers and return their sum"""
    c = a+b
    return {
        "result": c
    }

def Subtract(a: int, b: int)-> dict:
    """Subtract 2 integers and return their difference"""
    c = a-b
    return {
        "result": c
    }

def Multiply(a: int, b: int)-> dict:
    """Multiply 2 integers and return their product"""
    c = a*b
    return {
        "result": c
    }

def Divide(a: int, b: int)-> dict:
    """Divide 2 integers and return their sum"""
    if b==0:
        return {
            "answer": "Undefined!"
        }
    c = a/b
    return {
        "result": c
    }

basic_math = Agent(
    name="basic_math_agent",
    model="gemini-2.5-flash-lite",
    description="Thiis agent solves basic arithmetic problems.",
    instruction="""
    You are a helpful AI assistant tasked with solving simple math sums (addition, subtraction, multiplication, division).
    You have 4 tools at your disposal, one for each operation.
    1. Add tool (for adding)
    2. Subtract tool (for subtracting)
    3. Multiply tool (for multiplying)
    4. Divide tool (for dividing)
    RETURN THE RESULT after using the appropriate tool.

    {"result": the answer of the math sum asked.}
    
    NOTE: If a user gives another prompt after you are called, and it is unrelated to maths, you shall pass the responsibility 
    back to the Manager to decide who to delegate the user's request too. You should do this every time, without fail.
    """,
    tools=[Add, Subtract, Multiply, Divide]
)

# In the instruction section, it is clearly mentioned that if the prompt given is unrelated to maths, directly pass on the entire
# responsibility over to the manager agent to decide further flow. Very clear instructions given to the subagent here.

# Also, the return format is properly specified to ensure smooth result display.