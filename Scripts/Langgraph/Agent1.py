from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

