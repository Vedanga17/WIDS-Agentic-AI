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
- **RAG Pipeline**: Retrieval-augmented generation using ChromaDB vector store
- **Sentiment Analysis**: Multi-class sentiment classification using pre-trained BERT models
- **Graph-Based Agents**: Stateful agent workflows implemented with LangGraph
- **Vector Search**: Efficient semantic search over restaurant reviews using embeddings

## Project Structure

```
WIDS Project/
├── Scripts/
│   ├── lang_graph.py              # LangGraph stateful agent implementation
│   ├── local-ai-agent.py          # RAG-based Q&A system for restaurant reviews
│   ├── starting.py                # Sentiment analysis experiments
│   ├── vector.py                  # Vector store initialization and retrieval
│   └── realistic_restaurant_reviews.csv  # Restaurant review dataset
├── chroma_langchain_db/           # ChromaDB vector database storage
├── venv/                          # Python virtual environment
└── .gitignore                     # Git ignore configuration
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
   pip install langchain langchain-ollama langchain-chroma langchain-core
   pip install pandas transformers torch chromadb
   pip install langgraph ipython
   ```

## Usage

### Running the Restaurant Review Q&A Agent

This agent answers questions about a pizza restaurant based on customer reviews:

```bash
cd Scripts
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
python starting.py
```

### Exploring LangGraph Agents

Execute the stateful agent workflow:

```bash
cd Scripts
python lang_graph.py
```

## Components

### 1. Local AI Agent (`local-ai-agent.py`)

Implements a RAG-based question-answering system that:
- Retrieves relevant restaurant reviews from the vector database
- Generates contextual answers using LLaMA 3.2
- Provides an interactive command-line interface

### 2. Vector Store (`vector.py`)

Handles:
- CSV data ingestion from restaurant reviews
- Document embedding using Ollama's mxbai-embed-large model
- ChromaDB vector store initialization and management
- Semantic similarity search with configurable retrieval parameters

### 3. Sentiment Analysis (`starting.py`)

Demonstrates:
- Pre-trained BERT-based sentiment classification
- Multi-class rating prediction (1-5 stars)
- Batch processing of review text

### 4. LangGraph Agent (`lang_graph.py`)

Showcases:
- Stateful agent workflows using LangGraph
- Graph-based agent architecture
- Node-based processing pipeline

## Technologies Used

- **LangChain**: Framework for developing LLM-powered applications
- **LangGraph**: Library for building stateful, multi-actor applications with LLMs
- **Ollama**: Local LLM deployment platform
- **ChromaDB**: Vector database for embeddings storage and retrieval
- **Transformers**: Hugging Face library for pre-trained NLP models
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
