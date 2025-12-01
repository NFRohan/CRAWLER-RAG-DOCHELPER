# 🕷️ CRAWLER-RAG-DOCHELPER

A powerful **Retrieval-Augmented Generation (RAG)** system that crawls documentation websites, ingests content into a vector database, and provides intelligent question-answering capabilities using LangChain and Google Gemini.

## 📋 Overview

This project automates the process of:
1. **Crawling** documentation websites using Tavily's advanced web crawling capabilities
2. **Processing** and chunking the crawled content for optimal retrieval
3. **Storing** document embeddings in Pinecone vector database
4. **Answering** questions using a RAG pipeline with Cohere reranking and Google Gemini LLM

## ✨ Features

- 🌐 **Intelligent Web Crawling** - Uses Tavily to crawl documentation sites with configurable depth and breadth
- 💾 **Local Caching** - Caches crawled data to avoid redundant API calls
- 📄 **Smart Text Splitting** - Recursive character splitting with configurable chunk sizes and overlap
- 🔍 **Semantic Search** - HuggingFace embeddings (`all-MiniLM-L12-v2`) for accurate semantic retrieval
- 🏆 **Reranking** - Cohere reranker for improved relevance scoring
- 🤖 **AI-Powered Answers** - Google Gemini 2.5 Flash for generating accurate, context-aware responses
- 📊 **Colorful Logging** - Beautiful console output with colored status messages

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Tavily Crawl  │────▶│  Text Splitter  │────▶│    Pinecone     │
│  (Web Crawling) │     │   (Chunking)    │     │ (Vector Store)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Google Gemini  │◀────│ Cohere Reranker │◀────│    Retriever    │
│      (LLM)      │     │   (Relevance)   │     │    (Search)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│     Answer      │
│   + Sources     │
└─────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Pipenv (for dependency management)
- API keys for:
  - [Tavily](https://tavily.com/) - Web crawling
  - [Pinecone](https://www.pinecone.io/) - Vector database
  - [Google AI](https://ai.google.dev/) - Gemini LLM
  - [Cohere](https://cohere.com/) - Reranking

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NFRohan/CRAWLER-RAG-DOCHELPER.git
   cd CRAWLER-RAG-DOCHELPER
   ```

2. **Install dependencies**
   ```bash
   pipenv install
   pipenv shell
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   TAVILY_API_KEY=your_tavily_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   GEMINI_API_KEY=your_gemini_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```

4. **Set up Pinecone**
   
   Create a Pinecone index named `doc-helper-index` with the appropriate dimensions for the embedding model (384 for `all-MiniLM-L12-v2`).

### Usage

#### 1. Ingest Documentation

Run the ingestion script to crawl and store documentation:

```bash
python ingestion.py
```

This will:
- Crawl the target documentation site (default: LangChain docs)
- Cache the crawled data locally in `crawled_data.json`
- Split documents into chunks (500 chars with 100 char overlap)
- Store embeddings in Pinecone

#### 2. Query the RAG System

Run the backend core to ask questions:

```bash
python backend/core.py
```

Example output:
```
Querying RAG chain with: 'What is the simplest way to get started with LangChain?'

--- Answer ---
[AI-generated answer based on documentation]

--- Sources ---
https://docs.langchain.com/...
```

## 📁 Project Structure

```
CRAWLER-RAG-DOCHELPER/
├── ingestion.py        # Web crawling and vector store ingestion
├── backend/
│   └── core.py         # RAG chain implementation and query handling
├── logger.py           # Colorful logging utilities
├── Pipfile             # Python dependencies
├── Pipfile.lock        # Locked dependencies
├── crawled_data.json   # Cached crawled data (generated)
├── LICENSE             # Apache 2.0 License
└── README.md           # This file
```

## 🔧 Configuration

### Crawling Parameters (ingestion.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 5 | Maximum depth for crawling |
| `max_breadth` | 20 | Maximum breadth for mapping |
| `max_pages` | 1000 | Maximum pages to crawl |
| `chunk_size` | 500 | Characters per chunk |
| `chunk_overlap` | 100 | Overlapping characters between chunks |

### RAG Parameters (backend/core.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 20 | Number of documents to retrieve |
| `temperature` | 0.2 | LLM temperature for response generation |
| `rerank_model` | `rerank-english-v2.0` | Cohere reranking model |

## 🛠️ Tech Stack

- **[LangChain](https://langchain.com/)** - Framework for building LLM applications
- **[Tavily](https://tavily.com/)** - AI-powered web crawling and search
- **[Pinecone](https://www.pinecone.io/)** - Vector database for semantic search
- **[HuggingFace](https://huggingface.co/)** - Sentence transformer embeddings
- **[Cohere](https://cohere.com/)** - Document reranking
- **[Google Gemini](https://ai.google.dev/)** - Large language model
- **[Pipenv](https://pipenv.pypa.io/)** - Python dependency management

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
  Made with ❤️ using LangChain and RAG
</p>
