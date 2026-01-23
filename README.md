# WIDS Agentic AI Project

A comprehensive exploration of agentic AI systems developed as part of the Winter in Data Science (WIDS) program at IIT Bombay. This project implements various AI agent architectures, including retrieval-augmented generation (RAG), sentiment analysis, and graph-based agent workflows using modern frameworks like LangChain and LangGraph.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Technologies Used](#technologies-used)
- [License](#license)

## Overview

This project demonstrates the implementation of intelligent AI agents capable of:
- Answering domain-specific questions using RAG (Retrieval-Augmented Generation)
- Performing sentiment analysis on textual data
- Building stateful agent workflows with LangGraph
- Managing vector embeddings with ChromaDB for efficient information retrieval

The primary use case involves a restaurant review analysis system that can answer questions about customer experiences using a combination of vector search and large language models.

## Features

- **Local AI Agent**: Question-answering system powered by Ollama's LLaMA 3.2 model
- **Google ADK Agents**: Implementation of agents using Google's Agent Development Kit (ADK) with Gemini models
- **RAG Pipeline**: Retrieval-augmented generation using ChromaDB vector store
- **PDF RAG Agent**: Advanced RAG agent that answers questions from PDF documents
- **ReAct Agent**: Reasoning and Acting agent with tool integration
- **Drafter Agent**: Automated document drafting assistant
- **Sentiment Analysis**: Multi-class sentiment classification using pre-trained BERT models
- **Graph-Based Agents**: Stateful agent workflows implemented with LangGraph
- **Vector Search**: Efficient semantic search over restaurant reviews and PDF documents using embeddings

## Project Structure

```
WIDS Project/
├── Scripts/
│   ├── transformer.py             # Sentiment analysis experiments
│   ├── ADK_Google/                # Google Agent Development Kit (ADK) implementations
│   │   ├── 1-Basic_Agent/
│   │   │   └── greeting_agent/    # Basic greeting agent with Gemini 2.5 Flash
│   │   ├── 2-Tool_Agent/
│   │   │   └── tool_agent/        # Agent with custom tools (factorial, current time)
│   │   └── 3-2nd_Agent/
│   │       └── wheel_fortunate_agent/  # Interactive fortune wheel game agent
│   ├── Langchain/
│   │   ├── local-ai-agent.py      # RAG-based Q&A system for restaurant reviews
│   │   ├── vector.py              # Vector store initialization and retrieval
│   │   └── realistic_restaurant_reviews.csv  # Restaurant review dataset
│   ├── Langgraph/
│   │   ├── Agents/                # LangGraph tutorial series
│   │   │   ├── lang_graph1.py     # Basic LangGraph structure
│   │   │   ├── lang_graph2.py     # Handling multiple inputs
│   │   │   ├── lang_graph3.py     # Multiple nodes and edges
│   │   │   ├── lang_graph4.py     # Conditional routing
│   │   │   └── lang_graph5.py     # Interactive number guessing game
│   │   └── AI Agents/             # Advanced AI agent implementations
│   │       ├── RAG_agent.py       # RAG agent with PDF document Q&A
│   │       ├── ReAct_agent.py     # ReAct (Reasoning + Acting) agent
│   │       ├── Drafter_agent.py   # Document drafting agent
│   │       ├── Agent1.py          # Basic agent implementation
│   │       └── leave_notification.txt  # Sample text document
│   └── chroma_langchain_db/       # ChromaDB vector database storage
├── venv/                          # Python virtual environment
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.8 or higher
- Ollama (for running local LLM models)
- Git

### Ollama Setup

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull required models:
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vedanga17/WIDS-Agentic-AI.git
   cd "WIDS Project"
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install langchain langchain-ollama langchain-chroma langchain-core
   pip install langchain-groq langchain-huggingface langchain-community
   pip install pandas transformers torch chromadb pypdf
   pip install langgraph ipython python-dotenv
   ```

4. (Optional) Install Google ADK for ADK agent examples:
   ```bash
   pip install google-adk
   ```

## Usage

### Running the Restaurant Review Q&A Agent

This agent answers questions about a pizza restaurant based on customer reviews:

```bash
cd Scripts/Langchain
python local-ai-agent.py
```

Example interaction:
```
Ask a question about the pizza restaurant (q to quit): What do customers say about the pizza quality?
```

### Testing Sentiment Analysis

Run sentiment analysis experiments on movie reviews:

```bash
cd Scripts
python transformer.py
```

### Running Google ADK Agents

The project includes implementations using Google's Agent Development Kit (ADK) with Gemini models:

#### 1. Basic Greeting Agent
```bash
cd Scripts/ADK_Google/1-Basic_Agent/greeting_agent
adk run
```
A simple conversational agent that:
- Greets users and asks for their name
- Provides personalized greetings using Gemini 2.5 Flash
- Demonstrates basic ADK agent setup

#### 2. Tool-Enabled Agent
```bash
cd Scripts/ADK_Google/2-Tool_Agent/tool_agent
adk run
```
An agent with custom tool integration that can:
- Calculate factorial of numbers
- Get current time in formatted output
- Demonstrates how to create and integrate custom Python functions as tools

#### 3. Wheel of Fortune Agent
```bash
cd Scripts/ADK_Google/3-2nd_Agent/wheel_fortunate_agent
adk run
```
An interactive game agent that:
- Spins a fortune wheel for random positive outcomes
- Displays prizes like free trips, cash, or celebrity meet-and-greets
- Shows how to handle random selection and explicit tool output

**Note:** These agents require Google ADK installation. Install with:
```bash
pip install google-adk
```

### Exploring LangGraph Tutorials

The project includes progressive LangGraph tutorials demonstrating various concepts:
/Agents
python lang_graph1.py
```
Learn the simplest LangGraph structure with a single node.

**2. Multiple Inputs (lang_graph2.py)**
```bash
python lang_graph2.py
```
Handle multiple inputs and perform calculations (addition/multiplication).

**3. Sequential Workflow (lang_graph3.py)**
```bash
python lang_graph3.py
```
Chain multiple nodes together in a sequential workflow.

**4. Conditional Routing (lang_graph4.py)**
```bash
python lang_graph4.py
```
Implement conditional logic to route between different operations.

**5. Interactive Game (lang_graph5.py)**
```bash
python lang_graph5.py
```
Build a complete interactive number guessing game with state management.

### Running Advanced AI Agents

**RAG Agent with PDF Documents**
```bash
cd Scripts/Langgraph/AI\ Agents
python RAG_agent.py
```
Ask questions about information contained in PDF documents. The agent retrieves relevant context from the PDF and generates accurate answers using RAG.

**ReAct Agent**
```bash
python ReAct_agent.py
```
Interact with an agent that uses the ReAct (Reasoning + Acting) framework to break down complex tasks, reason through problems, and take appropriate actions.

**Document Drafter Agent**
```bash
python Drafter_agent.py
```
Generate and draft professional documents with AI assistance.

**Basic Agent**
```bash
python Agent1.py
```
Explore fundamental agent architecture and implementation patterns
python lang_graph5.py
```
Build a complete interactive number guessing game with state management.
Agents/`)

Progressive tutorials showcasing LangGraph capabilities:

#### lang_graph1.py - Basic Structure
- Single node graph implementation
- Simple state management with TypedDict
- Entry and finish point configuration

#### lang_graph2.py - Multiple Inputs
- Processing lists of values
- Conditional operations (addition/multiplication)
- Handling complex input structures

#### lang_graph3.py - Sequential Workflow
- Multi-node pipeline with edges
- State transformation across nodes
- Sequential data processing

#### lang_graph4.py - Conditional Routing
- Dynamic node routing based on state
- Multiple conditional branches
- Decision-making functions

#### lang_graph5.py - Interactive Application
- Complete game implementation (number guessing)
- User interaction handling
- Complex state management with multiple attributes
- Iterative workflows with loop conditions

## Components

### 1. Google ADK Agents (`ADK_Google/`)

Google Agent Development Kit (ADK) implementations using Gemini models for various interactive tasks:

#### 1-Basic_Agent/greeting_agent
A foundational conversational agent demonstrating basic ADK setup and interaction patterns.

**Features:**
- Model: `gemini-2.5-flash`
- Greets users and collects their name
- Provides personalized greetings
- Clean agent architecture using ADK's Agent class
- Entry point to understanding Google ADK framework

**Use Case:** Simple chatbot that engages users with friendly conversation

#### 2-Tool_Agent/tool_agent
Advanced agent showcasing custom tool integration and multi-capability systems.

**Custom Tools:**
- `get_current_time()`: Returns formatted current timestamp
- `factorial()`: Calculates factorial of any given number

**Features:**
- Model: `gemini-2.5-flash`
- Demonstrates creating Python functions as agent tools
- Tool selection and execution handling
- Multi-tool architecture in single agent
- Shows how to extend agent capabilities with custom logic

**Use Case:** Utility assistant that can perform calculations and provide time information

#### 3-2nd_Agent/wheel_fortunate_agent
Interactive fortune wheel game demonstrating random selection and explicit tool output display.

**Custom Tools:**
- `fortunate_wheel()`: Randomly selects from three exciting outcomes:
  - Free trip to the Bahamas with coupons
  - $5000 cash prize
  - Celebrity meet-and-greet experience

**Features:**
- Model: `gemini-2.5-flash-lite` (optimized lightweight model)
- Random outcome generation using Python's random module
- Explicit instructions for tool result display
- Entertainment-focused interactive experience
- Critical output handling to ensure user sees results

**Use Case:** Fun interactive game that provides users with positive random outcomes

### 2. Sentiment Analysis (`transformer.py`)

Multi-class sentiment classification system for analyzing movie reviews and general text.

**Features:**
- Uses BERT-based pre-trained model: `nlptown/bert-base-multilingual-uncased-sentiment`
- 5-class sentiment classification (1-5 stars)
- Token classification and analysis
- PyTorch-based implementation
- HuggingFace Transformers integration

**Use Case:** Analyze customer reviews, social media sentiment, or any text-based feedback

### 3. LangGraph Tutorials (`Langgraph/Agents/`)

Progressive tutorials showcasing LangGraph capabilities:

#### lang_graph1.py - Basic Structure
- Single node graph implementation
- Simple state management with TypedDict
- Entry and finish point configuration

#### lang_graph2.py - Multiple Inputs
- Processing lists of values
- Conditional operations (addition/multiplication)
- Handling complex input structures

#### lang_graph3.py - Sequential Workflow
- Multi-node pipeline with edges
- State transformation across nodes
- Sequential data processing

#### lang_graph4.py - Conditional Routing
- Dynamic node routing based on state
- Multiple conditional branches
- Decision-making functions

#### lang_graph5.py - Interactive Application
- Complete game implementation (number guessing)
- User interaction handling
- Complex state management with multiple attributes
- Iterative workflows with loop conditions

### 4. Advanced AI Agents (`Langgraph/AI Agents/`)

Production-ready AI agent implementations:

#### RAG_agent.py - PDF Question-Answering Agent
- PDF document loading and processing with PyPDFLoader
- Text chunking with RecursiveCharacterTextSplitter
- Vector embeddings using HuggingFace models (all-MiniLM-L6-v2)
- ChromaDB integration for efficient document retrieval
- LangGraph-based agent workflow with tool integration
- Groq LLM integration (llama-3.3-70b-versatile)
- Custom retrieval tool for semantic search
- Error handling and validation for tool execution

#### ReAct_agent.py - Reasoning & Acting Agent
- ReAct framework implementation
- Multi-step reasoning process
- Tool selection and execution
- Observation-based decision making
- Chain-of-thought prompting

#### Drafter_agent.py - Document Drafting Agent
- Automated document generation
- Template-based content creation
- Context-aware writing assistance
- Professional document formatting

#### Agent1.py - Foundational Agent
- Core agent architecture patterns
- Basic state management
- Message handling and routing
- Simple tool integration example
ProgGroq**: High-performance LLM inference API
- **ChromaDB**: Vector database for embeddings storage and retrieval
- **HuggingFace**: Embedding models and transformers
Create a `.env` file in the project root for the AI Agents:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from [https://console.groq.com](https://console.groq.com)

#### lang_graph1.py - Basic Structure
- Single node graph implementation
- Simple state management with TypedDict
- Entry and finish point configuration

#### lang_graph2.py - Multiple Inputs
- Processing lists of values
- Conditional operations (addition/multiplication)
- Handling complex input structures

#### lang_graph3.py - Sequential Workflow
- Multi-node pipeline with edges
- State transformation across nodes
- Sequential data processing

#### lang_graph4.py - Conditional Routing
- Dynamic node routing based on state
- Multiple conditional branches
- Decision-making functions

#### lang_graph5.py - Interactive Application
- Complete game implementation (number guessing)
- User interaction handling
- Complex state management with multiple attributes
- Iterative workflows with loop conditions

## Technologies Used

- **LangChain**: Framework for developing LLM-powered applications
- **LangGraph**: Library for building stateful, multi-actor applications with LLMs
- **Google ADK**: Google's Agent Development Kit for building AI agents with Gemini models
- **Groq**: High-performance LLM inference API
- **ChromaDB**: Vector database for embeddings storage and retrieval
- **HuggingFace**: Embedding models and transformers library
- **Transformers**: Pre-trained NLP models for sentiment analysis
- **Pandas**: Data manipulation and analysis
- **PyTorch**: Deep learning framework

**Google ADK Agents:**
- LLM: `gemini-2.5-flash` and `gemini-2.5-flash-lite` (via Google ADK)
- Custom tool integration with Python functions
- Interactive agent workflows

**Local Agents:**
- LLM: `llama3.2` (via Ollama)
- Embeddings: `mxbai-embed-large` (via Ollama)

**Advanced AI Agents:**
- LLM: `llama-3.3-70b-versatile` (via Groq API)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFace)

**Sentiment Analysis:**
- ModelHugging Face library for pre-trained NLP models
- **Pandas**: Data manipulation and analysis
- **PyTorch**: Deep learning framework

## Configuration

### Environment Variables

No environment variables are currently required as the project uses local Ollama models.

### Model Configuration

Models are configured in the respective Python files:
- LLM: `llama3.2` (via Ollama)
- Embeddings: `mxbai-embed-large` (via Ollama)
- Sentiment Analysis: `nlptown/bert-base-multilingual-uncased-sentiment`

## Data

The project includes a sample dataset of restaurant reviews (`realistic_restaurant_reviews.csv`) with the following structure:
- Title: Review headline
- Review: Full review text
- Rating: Numerical rating
- Date: Review date

## License

This project is part of the IIT Bombay WIDS program. Please refer to your institution's guidelines for usage and distribution.

## Acknowledgments

- IIT Bombay Winter in Data Science (WIDS) Program
- LangChain and LangGraph communities
- Ollama project for local LLM deployment

---

**Author**: Vedanga  
**Institution**: IIT Bombay  
**Program**: Winter in Data Science (WIDS)  
**Year**: 2025-26
