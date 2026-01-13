# A number guessing game where the agent has to guess a randomly chosen number between 1 and 20.

from typing import TypedDict, List, Dict
import random
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict): # setting up the state
    player_name: str
    target: int
    guesses: List[int]
    attempts: int
    hint: str
    lower_bound: int
    upper_bound: int

def setup(state: AgentState) -> AgentState:
    """ Initalizes the game state with a random guess. """

    state['player_name'] = "Welcome, " + state['player_name'] + "!"
    state['target'] = random.randint(1, 20)
    state['guesses'] = []
    state['attempts'] = 0
    state['hint'] = "Hi! I'm thinking of a number between 1 and 20. Can you guess it?"
    state['lower_bound'] = 1
    state['upper_bound'] = 20
    return state

def process_guess(state: AgentState) -> AgentState:
    """ Processes the agent's guess and updates the state accordingly. """
    
    possible_guesses = [] # makes a list of possible guesses within the current bounds
    for i in range(state['lower_bound'], state['upper_bound'] + 1):
        if i not in state['guesses']:
            possible_guesses.append(i) # if not already guessed, add to possible guesses

    if possible_guesses:
        guess = random.choice(possible_guesses) # if already guessed, choose a new one from the possible_guesses
    else:
        guess = random.randint(state['lower_bound'], state['upper_bound']) # if not guessed, make a guess within bounds
        
    state["guesses"].append(guess) # appending the guess to the list to avoid repetition
    state["attempts"] += 1
    print("Attempt", state["attempts"], ": Current guess", guess, "Current bounds:", state['lower_bound'], "-", state['upper_bound'])
    return state

def hint_giver(state: AgentState) -> AgentState:
    """ Provides hints based on the agent's guess. """
    last_guess = state["guesses"][-1]
    target = state['target']
    
    if last_guess < target: # if guess is low, give hint and update lower bound
        state['hint'] = f"Your guess of {last_guess} is too low."
        state['lower_bound'] = max(state['lower_bound'], last_guess + 1)
        print("Hint: ", state['hint'])

    elif last_guess > target: # if guess is high, give hint and update upper bound
        state['hint'] = f"Your guess of {last_guess} is too high."
        state['upper_bound'] = min(state['upper_bound'], last_guess - 1)
        print("Hint: ", state['hint'])

    else:
        state['hint'] = f"Congratulations! You've guessed the number {target} in {state['attempts']} attempts!"
    
    return state

def checker(state: AgentState) -> str:
    """ Checks if the latest guess is correct. """
    
    latest_guess = state["guesses"][-1]
    if latest_guess == state['target']: # if correct guess, end the game
        print("Game Over: Correct guess!")
        return END
    elif state["attempts"] >= 7: # if attempt limit reached, end the game
        print("Game Over: Maximum attempts reached.")
        return END
    else:
        return "continue"

graph = StateGraph(AgentState)

graph.add_node("setup", setup)
graph.add_node("process_guess", process_guess)
graph.add_node("hint_giver", hint_giver) # creating the nodes

graph.set_entry_point("setup")
graph.add_edge("setup", "process_guess")
graph.add_edge("process_guess", "hint_giver") # adding connecting edges

graph.add_conditional_edges(
    "hint_giver", # source node
    checker, # action performed by the node
    {
        "continue": "process_guess", # edge: destination node
        END: END
    }
)

app = graph.compile()
input = AgentState(player_name="Student")

result = app.invoke(input)
print(result)