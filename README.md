# 🍳 Recipe RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system for answering questions about recipes. Built as a course project for the LLM/NLP course at Lviv Polytechnic National University.

## Features

- **Dual Retrieval Methods:**
  - **BM25** - Keyword-based search using Okapi BM25 algorithm
  - **Semantic Search** - Meaning-based search using sentence transformers
  - **Hybrid Mode** - Combines both methods with score normalization

- **Reranking:** Cross-encoder reranking for improved relevance

- **Metadata Filtering:** Filter recipes by category and ingredients

- **LLM-Agnostic:** Supports multiple LLM providers:
  - Groq (free tier available)
  - OpenRouter (free models available)
  - OpenAI
  - Ollama (local models)

- **Citations:** Answers include references to source recipes

- **Modern UI:** Gradio-based web interface

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio UI                               │
│  [Query Input] [API Key] [Toggles] [Answer + Citations]     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAG Pipeline                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Metadata    │→ │  Retriever   │→ │  Reranker    │       │
│  │  Filter      │  │ (BM25+Dense) │  │(CrossEncoder)│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                              │               │
│                                              ▼               │
│                                      ┌──────────────┐       │
│                                      │   LiteLLM    │       │
│                                      │  (LLM Call)  │       │
│                                      └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd rag-lpnlp-2025
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare dataset

Place your `full_dataset.csv` file in the `data/` directory:

```bash
mkdir -p data
cp /path/to/full_dataset.csv data/
```

The dataset should have the following columns:
- `title` - Recipe name
- `ingredients` - JSON array of ingredients
- `directions` - JSON array of cooking steps
- `link` - Source URL
- `source` - Source name
- `NER` - Key ingredients (Named Entity Recognition)

## Usage

### Running locally

```bash
python app.py --data data/full_dataset.csv --max-recipes 5000
```

Options:
- `--data` - Path to CSV dataset (default: `data/full_dataset.csv`)
- `--max-recipes` - Maximum recipes to load (default: 5000)
- `--port` - Server port (default: 7860)
- `--share` - Create public Gradio share link

### Getting API Keys

**Groq (Recommended - Free):**
1. Go to https://console.groq.com/
2. Create an account
3. Generate API key

**OpenRouter (Free options available):**
1. Go to https://openrouter.ai/
2. Create an account
3. Generate API key

**Ollama (Local, no API key needed):**
1. Install Ollama: https://ollama.com/
2. Pull a model: `ollama pull llama3.2`
3. Select `ollama/llama3.2` in the UI

## Retrieval Method Comparison

### When BM25 works better:
- Specific ingredient searches: "recipe with cream cheese"
- Exact keyword matches: "chocolate chip cookies"
- When you know specific terms: "brisket marinade"

### When Semantic Search works better:
- Conceptual queries: "healthy dinner ideas"
- Natural language: "something sweet for a party"
- Synonym handling: "quick meal" → finds "fast", "easy" recipes

## Project Structure

```
rag-lpnlp-2025/
├── app.py                 # Gradio UI and main entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/
│   └── full_dataset.csv  # Recipe dataset
└── src/
    ├── __init__.py
    ├── chunking.py       # Data loading and chunking
    ├── retriever.py      # BM25 + Semantic retriever
    ├── reranker.py       # Cross-encoder reranker
    ├── metadata.py       # Metadata extraction and filtering
    ├── llm.py            # LLM wrapper (LiteLLM)
    └── rag_pipeline.py   # Main RAG pipeline
```

## Components Description

| Component | Description |
|-----------|-------------|
| **Data Source** | Recipe dataset (~2.2M recipes from cookbooks.com) |
| **Chunking** | Each recipe = one chunk (title + ingredients + directions) |
| **Retriever** | Hybrid BM25 + Semantic (sentence-transformers) |
| **Reranker** | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| **LLM** | LiteLLM wrapper for multi-provider support |
| **Metadata** | Category extraction from titles, ingredient filtering |
| **Citations** | Source references [1], [2] in answers |
| **UI** | Gradio with toggles and settings |

## Deployment to HuggingFace Spaces

1. Create a new Space on https://huggingface.co/spaces
2. Select "Gradio" as the SDK
3. Upload all files
4. Add `full_dataset.csv` to the `data/` folder
5. The app will auto-start

## Example Queries

```
"How do I make chocolate cake?"
"What's a quick chicken dinner recipe?"
"Dessert without nuts"
"Soup recipe with potatoes"
"Healthy breakfast ideas"
```

## Authors

- [Your Name] - LLM/NLP Course Project

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: Recipe collection from cookbooks.com
- Models: sentence-transformers, cross-encoder
- LLM: Powered by Groq/OpenRouter/OpenAI

