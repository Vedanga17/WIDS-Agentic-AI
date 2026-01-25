from google.adk.agents import Agent # importing the agent

question_answer_agent = Agent(
    name= "question_answer_agent",
    model= "gemini-2.5-flash-lite",
    description= "Question answering agent",
    instruction= """
    You are a helpful AI assistant. 
    You will be given a state, which will contain the name of a mathematician and his famous laws.
    The variables in which they are stored are:
    Name in {Mathematician} 
    Laws in {Famous_Formulae}
    You will answer the user's questions based on the provided information.
    You will ONLY return the name of the law specifically asked, not all of them.""",
)

# in instruction, the 2 variables passed are from basic_session_state.py.
# the agent, and its task, is defined here, and will be used for running basic_session_state.py.