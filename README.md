# WIDS Agentic AI Project

A comprehensive exploration of agentic AI systems developed as part of the Winter in Data Science (WIDS) program at IIT Bombay. This project implements various AI agent architectures, including Google ADK-based agents with Gemini models, retrieval-augmented generation (RAG), multi-agent hierarchical systems, transformer-based NLP tasks (sentiment analysis, text generation, summarization), and graph-based agent workflows using modern frameworks like Google ADK, LangChain, and LangGraph.

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

This project demonstrates the implementation of intelligent AI agents spanning multiple frameworks and paradigms:

**Google ADK Agent Architectures:**
- Basic conversational agents with Gemini 2.5 Flash models
- Tool-enabled agents with custom Python functions (factorial, time retrieval)
- Structured output agents using Pydantic schemas for JSON validation
- Session-based agents with state management and contextual information
- Multi-agent hierarchical systems with intelligent delegation and coordination
- Real-time financial data integration with yfinance for stock price retrieval

**LangChain & LangGraph Implementations:**
- Retrieval-Augmented Generation (RAG) for domain-specific question answering
- PDF document processing and question answering with vector search
- ReAct (Reasoning + Acting) agents with mathematical tool integration
- Document drafting agents with iterative refinement capabilities
- Router agents with domain expert specialization
- Stateful multi-step workflows with conditional routing
- Interactive conversational agents with message history

**Transformer-Based NLP Tasks:**
- Multi-class sentiment analysis using BERT (5-star rating classification)
- Creative text generation with GPT-2 (autoregressive continuation)
- Abstractive text summarization with BART (intelligent condensation)

**Vector Search & Embeddings:**
- ChromaDB integration for semantic document retrieval
- Restaurant review analysis system with top-k retrieval
- Efficient embedding management and persistent vector stores

The project progresses from foundational agent concepts to sophisticated multi-agent systems, demonstrating practical applications including customer review analysis, biographical information retrieval, real-time financial queries, and automated document generation.

## Features

- **Local AI Agent**: Question-answering system powered by Ollama's LLaMA 3.2 model
- **Google ADK Agents**: Implementation of agents using Google's Agent Development Kit (ADK) with Gemini models
  - Basic conversational agents
  - Tool-enabled agents with custom functions
  - Structured output agents with JSON schemas
  - Session-based state management agents
  - Multi-agent hierarchical systems with delegation
- **RAG Pipeline**: Retrieval-augmented generation using ChromaDB vector store
- **PDF RAG Agent**: Advanced RAG agent that answers questions from PDF documents
- **ReAct Agent**: Reasoning and Acting agent with tool integration
- **Drafter Agent**: Automated document drafting assistant
- **Sentiment Analysis**: Multi-class sentiment classification using pre-trained BERT models
- **Text Generation**: GPT-2 based text continuation and creative writing
- **Text Summarization**: BART-based abstractive summarization
- **Graph-Based Agents**: Stateful agent workflows implemented with LangGraph
- **Vector Search**: Efficient semantic search over restaurant reviews and PDF documents using embeddings
- **Stock Price Integration**: Real-time financial data retrieval using yfinance

## Project Structure

```
WIDS Project/
├── Scripts/
│   ├── transformer/               # Transformer-based NLP tasks
│   │   └── Assignment_1/          # Transformer pipeline implementations
│   │       ├── sentiment.py       # Sentiment analysis with BERT
│   │       ├── text_gen.py        # Text generation with GPT-2
│   │       ├── summarization.py   # Text summarization with BART
│   │       └── overall_pipeline.py # Combined NLP pipeline
│   ├── ADK_Google/                # Google Agent Development Kit (ADK) implementations
│   │   ├── 1-Basic_Agent/
│   │   │   └── greeting_agent/    # Basic greeting agent with Gemini 2.5 Flash
│   │   ├── 2-Tool_Agent/
│   │   │   └── tool_agent/        # Agent with custom tools (factorial, current time)
│   │   ├── 3-2nd_Agent/
│   │   │   └── wheel_fortunate_agent/  # Interactive fortune wheel game agent
│   │   ├── 4-Structured_Agent/
│   │   │   └── paragraph_agent/   # Agent with structured JSON output
│   │   ├── 5-Sessions_Based_Agent/
│   │   │   ├── question_answer_agent/  # Agent with session state management
│   │   │   └── basic_session_state.py  # Session state demonstration
│   │   └── 6-Multi_Agent_Based/
│   │       └── manager_agent/     # Hierarchical multi-agent system
│   │           ├── agent.py       # Manager agent coordinator
│   │           └── sub_agents/    # Specialized sub-agents
│   │               ├── basic_math/    # Mathematical operations agent
│   │               └── DOB_giver/     # Date of birth lookup agent
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
│   │       ├── leave_notification.txt  # Sample text document
│   │       └── Assignment_2/      # Assignment implementations
│   │           ├── Assn2_Q1.py    # Conversational agent with message history
│   │           ├── Assn2_Q2.py    # Two-step analyzer-generator agent
│   │           └── Assn2_Q3.py    # Router agent with domain experts
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
   pip install langgraph ipython python-dotenv google-adk yfinance
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

#### 4. Structured Output Agent
```bash
cd Scripts/ADK_Google/4-Structured_Agent/paragraph_agent
adk run
```
An agent demonstrating structured JSON output using Pydantic schemas:
- Generates ~50 word paragraphs on user-specified topics
- Enforces output format with Pydantic BaseModel
- JSON-only responses with field validation
- Word count constraints (45-65 words)
- Demonstrates output_schema and output_key parameters

#### 5. Session-Based Agent
```bash
cd Scripts/ADK_Google/5-Sessions_Based_Agent
python basic_session_state.py
```
An agent showcasing session state management:
- Pre-loaded state with mathematician information
- Contextual question answering based on session state
- State variables: mathematician name and famous formulae
- Demonstrates agent-session separation architecture
- Selective output based on user queries

#### 6. Multi-Agent Manager System
```bash
cd Scripts/ADK_Google/6-Multi_Agent_Based/manager_agent
adk web
```
A sophisticated hierarchical agent system with delegation and coordination:

**Manager Agent:**
- Intelligently routes requests to specialized sub-agents
- Uses custom tools for direct operations
- Synthesizes and formats responses from sub-agents
- Handles multi-domain queries in single conversation

**Sub-Agents:**
- **basic_math**: Arithmetic operations (add, subtract, multiply, divide)
- **DOB_giver**: Celebrity birth date lookup using Google Search

**Custom Tools:**
- **current_stock_price**: Real-time stock price retrieval using yfinance
  - Fetches latest closing prices
  - Supports any stock ticker symbol (e.g., AAPL, GOOGL, TSLA)
  - Returns formatted price with 2 decimal precision

**Features:**
- Automatic function calling (AFC) with up to 10 remote calls
- Hierarchical delegation patterns
- Responsibility handoff between agents
- Mixed tool types (custom functions + built-in Google Search)
- Sub-agents manage their own tool ecosystems

**Use Case:** Universal assistant handling math, biographical queries, and financial data in coordinated multi-agent workflow

**Note:** Requires yfinance: `pip install yfinance`

### Transformer-Based NLP Tasks

The project includes comprehensive transformer pipeline implementations:

#### Sentiment Analysis
```bash
cd Scripts/transformer/Assignment_1
python sentiment.py
```
Multi-class sentiment analysis using BERT:
- Model: `nlptown/bert-base-multilingual-uncased-sentiment`
- 5-star rating classification for movie reviews
- Confidence scores for each prediction
- Batch processing of multiple reviews

#### Text Generation
```bash
python text_gen.py
```
Creative text continuation using GPT-2:
- Generates coherent text from user prompts
- Multiple sequence generation (2 variations)
- Configurable max length (100 tokens) and new tokens (50)
- Demonstrates autoregressive text generation

#### Text Summarization
```bash
python summarization.py
```
Abstractive summarization using BART:
- Model: `facebook/bart-large-cnn`
- Configurable summary length (60-150 tokens)
- Compression ratio analysis
- Word count comparison between original and summary

**Note:** First-time runs will download model weights from HuggingFace (may take time depending on connection)

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
Explore fundamental agent architecture and implementation patterns.

### Running Assignment 2 Solutions

These assignments demonstrate practical LangGraph implementations:

**Assignment 2 Question 1 - Conversational Agent**
```bash
cd Scripts/Langgraph/AI\ Agents/Assignment_2
python Assn2_Q1.py
```
Interact with a conversational agent that maintains context across multiple exchanges. Ask questions about math, coding, or general knowledge.

**Assignment 2 Question 2 - Analyzer-Generator Pipeline**
```bash
python Assn2_Q2.py
```
Watch as your complex questions are first simplified by an analyzer agent, then answered by a generator agent in a two-step process.

**Assignment 2 Question 3 - Smart Router Agent**
```bash
python Assn2_Q3.py
```
Ask Python programming or general knowledge questions - the router intelligently directs your query to the appropriate expert agent.

**Note:** All Assignment 2 agents require a Groq API key in your `.env` file.

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

#### 4-Structured_Agent/paragraph_agent
Agent demonstrating structured JSON output using Pydantic schemas for validated responses.

**Features:**
- Model: `gemini-2.5-flash-lite`
- Pydantic BaseModel for output schema definition
- Field-level descriptions for structured data
- JSON-only response enforcement
- Word count constraints (45-65 words per paragraph)
- Uses LlmAgent class with output_schema parameter
- Output key specification for state storage

**Technical Details:**
- Output schema defined using Pydantic Field with descriptions
- Instructions explicitly mandate JSON format
- No extraneous output beyond JSON structure
- Demonstrates type-safe, validated agent responses

**Use Case:** Generating structured content with guaranteed format compliance for downstream processing

#### 5-Sessions_Based_Agent/question_answer_agent
Agent showcasing session state management and contextual information retrieval.

**Features:**
- Model: `gemini-2.5-flash-lite`
- Pre-loaded session state with contextual information
- State variables: Mathematician name and famous formulae
- Selective information retrieval based on queries
- Demonstrates agent-session architecture separation
- Answers questions using only provided state context

**Architecture:**
- Agent definition in `question_answer_agent/agent.py`
- Session management in `basic_session_state.py`
- State injection into agent context
- Targeted response generation (only requested information)

**Use Case:** Context-aware Q&A systems where responses depend on pre-loaded session-specific data

#### 6-Multi_Agent_Based/manager_agent
Sophisticated hierarchical multi-agent system demonstrating agent coordination, delegation, and tool orchestration.

**System Architecture:**

**Manager Agent (Root Agent):**
- Model: `gemini-2.5-flash-lite`
- Analyzes user queries and routes to appropriate handler
- Delegates to specialized sub-agents or invokes direct tools
- Synthesizes results from sub-agents/tools
- Formats user-friendly responses
- Handles multi-turn conversations across domains

**Sub-Agents:**

1. **basic_math** (Arithmetic Specialist):
   - Four custom tools: Add, Subtract, Multiply, Divide
   - Integer operations with result dictionaries
   - Zero-division handling in Divide
   - Returns structured {"result": value} responses
   - Hands back control for non-math queries

2. **DOB_giver** (Biographical Information Specialist):
   - Integrated Google Search tool for web queries
   - Finds birth dates of famous personalities
   - Structured output: {"name": str, "dob": DD/MM/YYYY}
   - Web search strategy for biographical data
   - Returns control for non-biographical queries

**Direct Tools:**

- **current_stock_price(stock_name)**: 
  - Integration with yfinance library
  - Fetches real-time stock prices
  - Retrieves 1-day historical data
  - Extracts latest closing price
  - Returns: {"Stock Price": "formatted price with 2 decimals"}
  - Supports any valid ticker symbol (AAPL, GOOGL, TSLA, etc.)

**Key Technical Features:**
- Automatic Function Calling (AFC) enabled with max 10 remote calls
- Tool chaining and nested agent delegation
- Mixed tool types: custom functions + built-in Google Search
- Sub-agents manage their own tool ecosystems independently
- Hierarchical delegation with responsibility handoff
- Explicit response instructions to prevent silent execution

**Critical Implementation Details:**
- Root agent variable naming convention (ADK requirement)
- Proper import paths for sub-agent packages: `.sub_agents.agent_name.agent`
- Tool import specificity: `from google.adk.tools.google_search import google_search`
- Sub-agents in `sub_agents` list, not wrapped in AgentTool
- Tool vs. agent distinction: simple functions vs. complex multi-tool agents

**Use Case:** Universal intelligent assistant handling mathematical calculations, biographical queries, and real-time financial data through coordinated multi-agent delegation

**Dependencies:** `pip install yfinance google-adk`

### 2. Transformer-Based NLP (`transformer/Assignment_1/`)

Comprehensive transformer pipeline implementations using HuggingFace pre-trained models:

#### sentiment.py - Multi-Class Sentiment Analysis
Advanced sentiment classification system using BERT-based models.

**Features:**
- Model: `nlptown/bert-base-multilingual-uncased-sentiment`
- 5-class sentiment classification (1-5 star ratings)
- Batch processing of multiple movie reviews
- Confidence score output (4 decimal precision)
- Multilingual support (trained on multiple languages)
- HuggingFace pipeline API for easy integration

**Technical Details:**
- BERT-based architecture for contextual understanding
- Pre-trained on multilingual sentiment data
- Token classification with softmax confidence scores
- Handles 5 reviews in single batch

**Use Case:** Analyze customer reviews, social media sentiment, product feedback, or any text-based ratings

#### text_gen.py - Creative Text Generation
Autoregressive text generation using GPT-2 for coherent continuations.

**Features:**
- Model: `gpt2` (124M parameters)
- Generates creative text from user-provided prompts
- Multiple sequence generation (2 variations per input)
- Configurable parameters:
  - max_length: 100 tokens (total)
  - max_new_tokens: 50 (newly generated)
  - num_return_sequences: 2 (variations)
  - truncation: True (handles long inputs)

**Technical Details:**
- Autoregressive generation (predicts next token iteratively)
- Transformer decoder architecture
- Temperature and top-k/top-p sampling for diversity
- Multiple outputs for creative variation

**Use Case:** Content creation, story writing, prompt completion, creative brainstorming, writing assistance

#### summarization.py - Abstractive Text Summarization
Intelligent text summarization using BART for information compression.

**Features:**
- Model: `facebook/bart-large-cnn` (406M parameters)
- Abstractive summarization (generates new sentences, not extraction)
- Configurable summary length:
  - max_length: 150 tokens
  - min_length: 60 tokens
- Truncation handling for long documents
- Word count comparison (original vs. summary)
- Compression ratio analysis

**Technical Details:**
- BART: Bidirectional and Auto-Regressive Transformer
- Encoder-decoder architecture
- Pre-trained on CNN/DailyMail dataset
- Generates fluent, coherent summaries

**Use Case:** Document summarization, article condensation, meeting notes, research paper abstracts, content briefs

**overall_pipeline.py - Integrated NLP Pipeline**
Combined pipeline integrating all three transformer tasks for comprehensive text analysis.

**Note:** First-time execution downloads pre-trained model weights from HuggingFace (several GB, time varies by connection speed)

### 2. Sentiment Analysis (`transformer.py`)

Multi-class sentiment classification system for analyzing movie reviews and general text.

**Features:**
- Uses BERT-based pre-trained model: `nlptown/bert-base-multilingual-uncased-sentiment`
- 5-class sentiment classification (1-5 stars)
- Token classification and analysis
- PyTorch-based implementation
- HuggingFace Transformers integration

**Note:** This file is legacy. See `transformer/Assignment_1/sentiment.py` for the current implementation.

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

#### Assignment_2/ - LangGraph Assignment Solutions

Practical assignments demonstrating progressive LangGraph concepts:

**Assn2_Q1.py - Conversational Agent with Context**
- Simple LangGraph application with Groq LLM (llama-3.3-70b-versatile)
- Maintains conversation history for contextual responses
- Continuous chat loop with message accumulation
- Handles multi-turn conversations with AI and human messages
- Expert in math, coding, and general knowledge domains

**Key Features:**
- Message history tracking (HumanMessage and AIMessage)
- State management with TypedDict
- Context-aware responses using accumulated chat history
- Interactive loop until user exits

**Assn2_Q2.py - Two-Step Analyzer-Generator Pipeline**
- Multi-agent workflow with question analysis and answer generation
- Sequential processing: Analyzer → Generator
- Demonstrates agent chaining and state passing

**Architecture:**
1. **Question Analyzer**: Simplifies complex questions while retaining meaning
2. **Answer Generator**: Provides detailed answers to simplified questions

**Key Features:**
- SystemMessage integration for agent instructions
- State propagation between agents
- Question rewriting without answering
- Specialized agents with distinct roles

**Assn2_Q3.py - Intelligent Router with Domain Experts**
- Conditional routing based on question classification
- LLM-powered decision making for expert selection
- Demonstrates dynamic workflow routing

**Architecture:**
1. **Router Agent**: Classifies questions using LLM (Python vs General)
2. **Python Expert**: Handles Python programming questions
3. **General Expert**: Handles general knowledge questions

**Key Features:**
- Intelligent LLM-based question classification
- Conditional edges with `decide_expert` function
- Domain-specific expert agents
- Dynamic routing to appropriate specialist
- START and END node usage

**Use Case:** Demonstrates practical LangGraph patterns for building multi-agent systems with routing logic

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
- **Transformers**: Pre-trained NLP models (BERT, GPT-2, BART)
- **yfinance**: Real-time financial market data retrieval
- **Pandas**: Data manipulation and analysis
- **PyTorch**: Deep learning framework
- **Pydantic**: Data validation and settings management using Python type annotations

**Google ADK Agents:**
- LLM: `gemini-2.5-flash` and `gemini-2.5-flash-lite` (via Google ADK)
- Custom tool integration with Python functions
- Interactive agent workflows
- Built-in tools: Google Search

**Local Agents:**
- LLM: `llama3.2` (via Ollama)
- Embeddings: `mxbai-embed-large` (via Ollama)

**Advanced AI Agents:**
- LLM: `llama-3.3-70b-versatile` (via Groq API)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFace)

**Transformer Models:**
- Sentiment Analysis: `nlptown/bert-base-multilingual-uncased-sentiment`
- Text Generation: `gpt2` (124M parameters)
- Summarization: `facebook/bart-large-cnn` (406M parameters)Hugging Face library for pre-trained NLP models
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
