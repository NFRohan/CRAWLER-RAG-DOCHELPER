# 🕷️ CRAWLER-RAG-DOCHELPER

A powerful **Retrieval-Augmented Generation (RAG)** system with **Hybrid Search** that crawls documentation websites, ingests content into a vector database, and provides intelligent question-answering capabilities using LangChain and Google Gemini.

## 📋 Overview

This project automates the process of:
1. **Fetching** documentation URLs from `llms.txt` index files
2. **Extracting** content from each page using Tavily Extract API
3. **Processing** and chunking the content for optimal retrieval
4. **Storing** both **Dense** (semantic) and **Sparse** (BM25 keyword) vectors in Pinecone
5. **Answering** questions using a **Hybrid Search RAG pipeline** with Cohere reranking and Google Gemini LLM

## ✨ Features

- 📑 **llms.txt Support** - Parses standard `llms.txt` documentation index files for targeted crawling
- 🌐 **Intelligent Content Extraction** - Uses Tavily Extract API to pull clean content from documentation pages
- 🔗 **URL Filtering** - Configurable regex patterns to focus on specific documentation sections
- 💾 **Multi-level Caching** - Caches both URL lists and extracted content to minimize API calls
- 📄 **Smart Text Splitting** - Recursive character splitting with configurable chunk sizes and overlap
- 🔍 **Hybrid Search (Sparse-Dense)** - Combines BM25 keyword matching with semantic embeddings for superior retrieval
- 🎯 **Dense Vectors** - HuggingFace embeddings (`all-MiniLM-L12-v2`) for semantic understanding
- 📊 **Sparse Vectors** - BM25 encoding for precise keyword matching
- 🏆 **Reranking** - Cohere reranker for improved relevance scoring
- 🤖 **AI-Powered Answers** - Google Gemini 2.5 Flash for generating accurate, context-aware responses
- 📊 **Colorful Logging** - Beautiful console output with colored status messages and error tracking

## 🏗️ Architecture

```
                            ┌─────────────────────────────────────────────────────────────┐
                            │                     INGESTION PIPELINE                      │
                            └─────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   llms.txt      │────▶│   URL Filter    │────▶│  Tavily Extract │────▶│  Text Splitter  │
│  (Fetch Index)  │     │ (Regex Pattern) │     │ (Content Fetch) │     │   (Chunking)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                 │
                        ┌────────────────────────────────────────────────────────┤
                        │                                                        │
                        ▼                                                        ▼
               ┌─────────────────┐                                      ┌─────────────────┐
               │  BM25 Encoder   │                                      │   HuggingFace   │
               │ (Sparse Vector) │                                      │ (Dense Vector)  │
               └────────┬────────┘                                      └────────┬────────┘
                        │                                                        │
                        └────────────────────────┬───────────────────────────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │    Pinecone     │
                                        │ (Hybrid Index)  │
                                        │  dotproduct     │
                                        └────────┬────────┘
                                                 │
                            ┌────────────────────────────────────────────────────────────┐
                            │                      QUERY PIPELINE                        │
                            └────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │   User Query    │
                                        └────────┬────────┘
                                                 │
                        ┌────────────────────────┴───────────────────────────────┐
                        │                                                        │
                        ▼                                                        ▼
               ┌─────────────────┐                                      ┌─────────────────┐
               │  BM25 Encoder   │                                      │   HuggingFace   │
               │ (Sparse Query)  │                                      │ (Dense Query)   │
               └────────┬────────┘                                      └────────┬────────┘
                        │                                                        │
                        └────────────────────────┬───────────────────────────────┘
                                                 │
                                                 ▼
                                   ┌───────────────────────────┐
                                   │      Hybrid Search        │
                                   │  α·Dense + (1-α)·Sparse   │
                                   └─────────────┬─────────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │ Cohere Reranker │
                                        │  (Top Results)  │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │  Google Gemini  │
                                        │   (Generate)    │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │     Answer      │
                                        │   + Sources     │
                                        └─────────────────┘
```

## 🔀 Hybrid Search Explained

This project uses **Sparse-Dense Hybrid Search** to combine the best of both worlds:

| Vector Type | Method | Strength |
|-------------|--------|----------|
| **Dense** | Semantic Embeddings | Understanding meaning, synonyms, context |
| **Sparse** | BM25 Keyword Weights | Exact term matching, rare words, names |

The hybrid search uses an **alpha parameter** to balance between the two:
- `α = 1.0` → Pure semantic search (dense only)
- `α = 0.0` → Pure keyword search (sparse only)  
- `α = 0.5` → Balanced hybrid (default)

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- API keys for:
  - [Tavily](https://tavily.com/) - Content extraction
  - [Pinecone](https://www.pinecone.io/) - Vector database
  - [Google AI](https://ai.google.dev/) - Gemini LLM
  - [Cohere](https://cohere.com/) - Reranking

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NFRohan/CRAWLER-RAG-DOCHELPER.git
   cd CRAWLER-RAG-DOCHELPER
   ```

2. **Install uv (if not already installed)**
   ```bash
   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create virtual environment and install dependencies**
   ```bash
   uv venv
   uv sync
   ```

4. **Activate the virtual environment**
   ```bash
   # Windows (PowerShell)
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

5. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # Required
   PINECONE_API_KEY=your_pinecone_api_key
   TAVILY_API_KEY=your_tavily_api_key
   GEMINI_API_KEY=your_gemini_api_key
   COHERE_API_KEY=your_cohere_api_key
   
   # Optional
   PINECONE_INDEX=doc-helper-index
   PINECONE_REGION=us-east-1
   ```

6. **Set up Pinecone Index for Hybrid Search**
   
   Run the setup script to create a properly configured index:
   ```bash
   python setup_pinecone_index.py
   ```
   
   This creates an index with:
   - **Dimension**: 384 (for `all-MiniLM-L12-v2`)
   - **Metric**: `dotproduct` (required for hybrid search)
   - **Sparse vector support**: Enabled

### Usage

#### 1. Set Up Pinecone Index (First Time Only)

```bash
python setup_pinecone_index.py
```

Other commands:
```bash
python setup_pinecone_index.py stats   # View index statistics
python setup_pinecone_index.py delete  # Delete the index
```

#### 2. Ingest Documentation

Run the ingestion script to extract and store documentation:

```bash
python ingestion.py
```

This will:
- Fetch the `llms.txt` file from `https://docs.langchain.com/llms.txt`
- Parse and filter URLs (default: Python docs only)
- Extract content from each URL using Tavily (one at a time for reliability)
- Cache extracted content to `crawled_data.json`
- Split documents into chunks (500 chars with 100 char overlap)
- Train BM25 encoder and save parameters to `bm25_params.json`
- Generate both dense and sparse vectors for each chunk
- Upsert hybrid vectors to Pinecone

**URL Filtering**: Edit `URL_INCLUDE_PATTERNS` in `ingestion.py` to target specific sections:
```python
URL_INCLUDE_PATTERNS = [
    r"/oss/python/",      # Python docs (enabled by default)
    # r"/oss/javascript/", # Uncomment for JS docs
    # r"/langsmith/",      # Uncomment for LangSmith docs
]
```

**Re-run with fresh data**:
```bash
# Clear caches and re-extract
Remove-Item crawled_data.json, llms_txt_urls.json -ErrorAction SilentlyContinue
python ingestion.py
```

#### 3. Query the RAG System

Run the backend core to ask questions:

```bash
python backend/core.py
```

Example output:
```
============================================================
🔍 HYBRID SEARCH RAG SYSTEM
   Using Sparse-Dense vectors for keyword + semantic search
============================================================

Querying RAG chain with: 'What is the simplest way to get started with LangChain?'

--- Answer ---
[AI-generated answer based on documentation]

--- Sources ---
https://docs.langchain.com/...
```

## 📁 Project Structure

```
CRAWLER-RAG-DOCHELPER/
├── ingestion.py            # llms.txt parsing and hybrid vector ingestion
├── setup_pinecone_index.py # Pinecone index setup for hybrid search
├── setup_venv.ps1          # PowerShell script for venv setup
├── backend/
│   └── core.py             # Hybrid RAG chain implementation
├── logger.py               # Colorful logging utilities
├── pyproject.toml          # Python dependencies (uv)
├── uv.lock                 # Locked dependencies (generated)
├── .env                    # API keys (create this)
├── llms_txt_urls.json      # Cached URL list from llms.txt (generated)
├── crawled_data.json       # Cached extracted content (generated)
├── bm25_params.json        # BM25 encoder parameters (generated)
├── LICENSE                 # Apache 2.0 License
└── README.md               # This file
```

## 🔧 Configuration

### Ingestion Parameters (ingestion.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLMS_TXT_URL` | `https://docs.langchain.com/llms.txt` | Source llms.txt file |
| `URL_INCLUDE_PATTERNS` | `[r"/oss/python/"]` | Regex patterns to filter URLs |
| `MAX_URLS` | `None` | Limit URLs for testing (None = all) |
| `chunk_size` | 500 | Characters per chunk |
| `chunk_overlap` | 100 | Overlapping characters between chunks |
| `delay` | 0.5 | Seconds between API requests |

### Hybrid Search Parameters (backend/core.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 20 | Number of documents to retrieve |
| `alpha` | 0.5 | Hybrid balance (0=sparse, 1=dense) |
| `temperature` | 0.3 | LLM temperature for response generation |
| `rerank_model` | `rerank-english-v3.0` | Cohere reranking model |

### Pinecone Index Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `index_name` | `doc-helper-index` | Name of the Pinecone index |
| `dimension` | 384 | Vector dimension (all-MiniLM-L12-v2) |
| `metric` | `dotproduct` | Required for hybrid search |
| `cloud` | `aws` | Cloud provider |
| `region` | `us-east-1` | Deployment region |

## 🛠️ Tech Stack

- **[LangChain](https://langchain.com/)** - Framework for building LLM applications
- **[Tavily](https://tavily.com/)** - AI-powered content extraction
- **[Pinecone](https://www.pinecone.io/)** - Vector database with hybrid search support
- **[pinecone-text](https://github.com/pinecone-io/pinecone-text)** - BM25 sparse encoding
- **[HuggingFace](https://huggingface.co/)** - Sentence transformer embeddings
- **[Cohere](https://cohere.com/)** - Document reranking
- **[Google Gemini](https://ai.google.dev/)** - Large language model
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package management

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**NFRohan** - [GitHub Profile](https://github.com/NFRohan)

---

<p align="center">
  Made with ❤️ using LangChain, Hybrid Search, and RAG
</p>
