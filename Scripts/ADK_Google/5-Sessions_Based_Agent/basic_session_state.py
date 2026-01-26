import uuid # for generating unique, random session IDs
import asyncio # for async programming (handling network delays, waiting, etc); I/O operations involved (get_session, create_session)

from dotenv import load_dotenv
from google.adk.sessions import InMemorySessionService # for creating a session within ADK
from google.genai import types
from google.adk.runners import Runner # the 'runner' combines: session and agent(s), critical for ACTUALLY running the code
from question_answer_agent import question_answer_agent # importing agent defined in agent.py

load_dotenv() # for storing the API key

# using async to handle waiting for session creation (while session is being created, perform some other task, and then come back when ready)
async def main():
    # Create a new session service to store the state (class instance)
    session_service_new = InMemorySessionService()

    # define an initial state to work with
    initial_state = {
    "Mathematician": "Isaac Newton",
        "Famous_Formulae": """
        1. Newton's Laws of motion (set of 3 laws in classical mechanics).
        2. Newton's Law of Gravitation (fundamental law of gravity).
        3. Newton's Method (core numerical analysis formula).
        4. Newton's Law of Cooling (thermodynamics).
    """,
        }

    # app name, user id, and session id stored in 3 variables
    APP_NAME = "Newton Info Bot"
    USER_ID = "Vedanga Gupta"
    SESSION_ID = str(uuid.uuid4()) # generates a long, unique sequence of alphanumeric characters

# creating a new session, and passing in the required inputs through pre-defined variables
    session = await session_service_new.create_session(   # have to use "await" here because a session is being created (meanwhile, the 
                                                          # compiler will perform some other task, and then return here)
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state,
    )

    print("Created a new session!")
    print(f"\tSession ID: {SESSION_ID}")

# defining the runner, and passing in the required inputs, as well as the session service
    runner = Runner(
    agent=question_answer_agent,
    app_name=APP_NAME,
    session_service=session_service_new,
    )

    question = input("Ask your question about Newton's famous laws:")

# storing the question and its contents in a variable to be passed to the runner

    message = types.Content(
        role="user", parts=[types.Part(text=question)]
    )

# running all events in the runner (in this case, only 1)
    for event in runner.run(
    user_id=USER_ID,
    session_id=SESSION_ID,
    new_message=message,
):
        if event.is_final_response(): # if it's the final event (and thus response), print the final response and its contents, if not empty
            if event.content and event.content.parts:
                print(f"Final Response: {event.content.parts[0].text}")

    print("==== Session Event Exploration ====")
    session_ = await session_service_new.get_session( # used "await" here to handle waiting for the session_service to fetch the relevant session here
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    # Log (and print) final Session state
    print("=== Final Session State ===")
    for key, value in session_.state.items():
        print(f"{key}: {value}")

# function to run the code, wrapped inside the async main.
if __name__ == "__main__":
    asyncio.run(main())

