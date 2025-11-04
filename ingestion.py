import asyncio
import os
import ssl
import json  # <-- Import the json module
from typing import Any, Dict, List, Optional
import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import log_error, log_info, log_warning, log_success, log_header
# from langchain_huggingface import HuggingFaceEmbeddings # <-- This is a duplicate import

load_dotenv()

# ssl with certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


tavily_extract = TavilyExtract()
travily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
travily_crawl = TavilyCrawl()

# embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={"device": "cpu"})

# --- Caching Logic: Define the cache file path ---
CACHE_FILE = "crawled_data.json"


async def main():
    """Main function to run the ingestion process."""
    log_header("Starting Ingestion Process")

    crawled_data = None  # Initialize variable

    # --- Caching Logic: Check if cache file exists ---
    if os.path.exists(CACHE_FILE):
        log_info(f"Found local cache. Loading data from {CACHE_FILE}...")
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                crawled_data = json.load(f)
            log_success(
                f"Successfully loaded {len(crawled_data.get('results', []))} pages from cache.")
        except json.JSONDecodeError:
            log_warning(
                f"Cache file {CACHE_FILE} is corrupted. Re-crawling...")
            crawled_data = None  # Reset to ensure re-crawl
        except Exception as e:
            log_error(f"Error reading cache file: {e}. Re-crawling...")
            crawled_data = None  # Reset to ensure re-crawl

    # --- Caching Logic: If cache is empty or invalid, crawl and save ---
    if crawled_data is None:
        log_info("No local cache found (or cache was invalid). Crawling website...")
        seed_url = "https://docs.langchain.com/oss/python/langchain/overview"
        crawled_data = travily_crawl.invoke({
            "url": seed_url,
            "max_depth": 5,
            "extract_depth": "advanced",
        })
        log_success(f"Crawled {len(crawled_data['results'])} pages.")

        # Save the new data to the cache file
        log_info(f"Saving crawled data to {CACHE_FILE}...")
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(crawled_data, f, indent=4)
            log_success("Successfully saved data to cache.")
        except Exception as e:
            log_error(f"Error saving data to cache: {e}")
    # --- End of Caching Logic ---

    valid_crawled_data = [
        page for page in crawled_data["results"]
        if page.get("raw_content")
    ]
    
    print(
        f"✅ Found {len(valid_crawled_data)} pages with content out of {len(crawled_data['results'])} crawled.")
    if len(valid_crawled_data) < len(crawled_data["results"]):
        print("⚠️ Warning: Some pages were skipped due to missing content.")

    all_docs = [
        Document(
            page_content=page["raw_content"],
            metadata={"source": page["url"]}
        )
        for page in valid_crawled_data
    ]

    # split into chunks
    log_header("Splitting Documents into Chunks")
    log_info(f"Total documents before splitting: {len(all_docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(all_docs)
    log_success(f"Total documents after splitting: {len(split_docs)}")

    log_header("Storing Chunks in Vector Store")
    # store in vector store
    vectorstore = PineconeVectorStore.from_documents(
        documents=split_docs,
        index_name="doc-helper-index",
        embedding=embeddings
    )

    log_header("Ingestion Process Completed")
    log_info(f"Total chunks stored: {len(split_docs)}")
    log_info( # This line can error if split_docs is empty
        f"urls mapped: {list(set([doc.metadata['source'] for doc in split_docs]))}")


if __name__ == "__main__":
    asyncio.run(main())