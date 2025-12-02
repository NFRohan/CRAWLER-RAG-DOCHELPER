import asyncio
import os
import ssl
import json
import re
import time
import httpx
from typing import Any, Dict, List, Optional
import certifi
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from logger import log_error, log_info, log_warning, log_success, log_header
from tavily import TavilyClient

load_dotenv()

# ssl with certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Pinecone configuration
INDEX_NAME = os.getenv("PINECONE_INDEX", "doc-helper-index")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Tavily client for content extraction
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env file")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Dense embeddings model (384 dimensions)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# BM25 Sparse encoder for keyword-based search
bm25_encoder = BM25Encoder()

# --- Configuration ---
LLMS_TXT_URL = "https://docs.langchain.com/llms.txt"
LLMS_TXT_CACHE = "llms_txt_urls.json"
CACHE_FILE = "crawled_data.json"
BM25_PARAMS_FILE = "bm25_params.json"

# Filter patterns for URLs to include (regex patterns)
# Set to None to include all URLs, or provide patterns to filter
URL_INCLUDE_PATTERNS = [
    r"/oss/python/",  # Python docs
    # r"/oss/javascript/",  # Uncomment to include JS docs
    # r"/langsmith/",  # Uncomment to include LangSmith docs
]

# Maximum URLs to process (set to None for all)
MAX_URLS = None  # e.g., 100 for testing, None for all


def parse_llms_txt(content: str) -> List[Dict[str, str]]:
    """
    Parse llms.txt format and extract URLs with their descriptions.
    
    Format:
    - [Title](url): Description
    - [Title](url)
    """
    urls = []
    lines = content.strip().split('\n')
    
    # Regex to match markdown links: [text](url) or - [text](url): description
    link_pattern = re.compile(r'-\s*\[([^\]]*)\]\(([^)]+)\)(?::\s*(.*))?')
    
    for line in lines:
        match = link_pattern.match(line.strip())
        if match:
            title = match.group(1) or ""
            url = match.group(2)
            description = match.group(3) or ""
            
            # Skip null titles or index pages
            if title.lower() == "null" or not url:
                continue
                
            urls.append({
                "title": title.strip(),
                "url": url.strip(),
                "description": description.strip()
            })
    
    return urls


def filter_urls(urls: List[Dict[str, str]], patterns: List[str] = None) -> List[Dict[str, str]]:
    """Filter URLs based on include patterns."""
    if not patterns:
        return urls
    
    filtered = []
    for item in urls:
        for pattern in patterns:
            if re.search(pattern, item["url"]):
                filtered.append(item)
                break
    
    return filtered


async def fetch_llms_txt() -> List[Dict[str, str]]:
    """Fetch and parse llms.txt from LangChain docs."""
    log_header("Fetching LangChain Documentation Index")
    
    # Check cache first
    if os.path.exists(LLMS_TXT_CACHE):
        log_info(f"Loading cached URL list from {LLMS_TXT_CACHE}...")
        try:
            with open(LLMS_TXT_CACHE, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                log_success(f"Loaded {len(cached)} URLs from cache.")
                return cached
        except Exception as e:
            log_warning(f"Failed to load cache: {e}")
    
    log_info(f"Fetching {LLMS_TXT_URL}...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(LLMS_TXT_URL)
        response.raise_for_status()
        content = response.text
    
    urls = parse_llms_txt(content)
    log_success(f"Parsed {len(urls)} URLs from llms.txt")
    
    # Apply filters
    if URL_INCLUDE_PATTERNS:
        urls = filter_urls(urls, URL_INCLUDE_PATTERNS)
        log_info(f"After filtering: {len(urls)} URLs match patterns")
    
    # Apply max limit
    if MAX_URLS and len(urls) > MAX_URLS:
        urls = urls[:MAX_URLS]
        log_info(f"Limited to {MAX_URLS} URLs for processing")
    
    # Cache the URL list
    try:
        with open(LLMS_TXT_CACHE, 'w', encoding='utf-8') as f:
            json.dump(urls, f, indent=2)
        log_success(f"Cached URL list to {LLMS_TXT_CACHE}")
    except Exception as e:
        log_warning(f"Failed to cache URL list: {e}")
    
    return urls


def extract_content_single(urls: List[str], delay: float = 0.5) -> List[Dict[str, Any]]:
    """
    Extract content from URLs using Tavily Extract API.
    Process one URL at a time for maximum reliability.
    
    Args:
        urls: List of URLs to extract content from
        delay: Seconds to wait between requests to avoid rate limiting
    
    Returns:
        List of extraction results with 'url' and 'raw_content' keys
    """
    all_results = []
    failed_urls = []
    
    log_info(f"Extracting content from {len(urls)} URLs (one at a time)...")
    log_info(f"Delay between requests: {delay}s")
    
    for url in tqdm(urls, desc="Extracting content"):
        try:
            # Extract single URL
            response = tavily_client.extract(urls=[url])
            
            if response and "results" in response and len(response["results"]) > 0:
                result = response["results"][0]
                # Check if we got actual content
                if result.get("raw_content"):
                    all_results.extend(response["results"])
                else:
                    log_warning(f"Empty content: {url[:70]}")
                    failed_urls.append({"url": url, "reason": "empty_content"})
            else:
                log_warning(f"No results: {url[:70]}")
                failed_urls.append({"url": url, "reason": "no_results"})
            
        except Exception as e:
            error_msg = str(e)
            
            # Try to extract HTTP status code from error message
            status_code = None
            if "status" in error_msg.lower():
                import re
                match = re.search(r'(\d{3})', error_msg)
                if match:
                    status_code = match.group(1)
            
            # Log with status code if available
            if status_code:
                log_warning(f"HTTP {status_code}: {url[:60]}")
            else:
                log_warning(f"Failed: {url[:60]} - {error_msg[:80]}")
            
            failed_urls.append({"url": url, "reason": error_msg[:100], "status": status_code})
            
            # If rate limited, wait longer
            if "rate" in error_msg.lower() or "limit" in error_msg.lower() or status_code == "429":
                log_info("Rate limited, waiting 10 seconds...")
                time.sleep(10)
        
        # Delay between requests
        time.sleep(delay)
    
    # Summary
    log_info(f"Extraction complete: {len(all_results)} succeeded, {len(failed_urls)} failed")
    
    if failed_urls:
        log_warning("Failed URLs summary:")
        for item in failed_urls[:15]:
            if isinstance(item, dict):
                status = f"[{item.get('status', 'N/A')}]" if item.get('status') else ""
                log_warning(f"  {status} {item['url'][:70]} - {item.get('reason', 'unknown')[:40]}")
            else:
                log_warning(f"  {item}")
        if len(failed_urls) > 15:
            log_warning(f"  ... and {len(failed_urls) - 15} more")
    
    return all_results


def generate_sparse_vector(text: str, encoder: BM25Encoder) -> Dict[str, Any]:
    """
    Generate a sparse vector from text using BM25 encoding.

    Returns a dict with 'indices' and 'values' for Pinecone sparse vector format.
    """
    sparse_dict = encoder.encode_documents([text])[0]
    return {
        "indices": sparse_dict["indices"],
        "values": sparse_dict["values"]
    }


def hybrid_upsert_to_pinecone(
    documents: List[Document],
    dense_embeddings: List[List[float]],
    bm25_encoder: BM25Encoder,
    batch_size: int = 100
):
    """
    Upsert documents to Pinecone with both dense and sparse vectors.

    Args:
        documents: List of LangChain Document objects
        dense_embeddings: List of dense embedding vectors
        bm25_encoder: Fitted BM25 encoder for sparse vectors
        batch_size: Number of vectors to upsert per batch
    """
    log_info(f"Upserting {len(documents)} documents with hybrid vectors...")

    vectors = []
    for i, (doc, dense_emb) in enumerate(zip(documents, dense_embeddings)):
        # Generate sparse vector for this document
        sparse_vec = generate_sparse_vector(doc.page_content, bm25_encoder)

        vector_data = {
            "id": f"doc_{i}",
            "values": dense_emb,
            "sparse_values": sparse_vec,
            "metadata": {
                "source": doc.metadata.get("source", "unknown"),
                # Store first 1000 chars for retrieval
                "text": doc.page_content[:1000]
            }
        }
        vectors.append(vector_data)

    # Upsert in batches
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting batches", total=total_batches):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    log_success(
        f"Successfully upserted {len(vectors)} hybrid vectors to Pinecone!")


async def main():
    """Main function to run the ingestion process."""
    log_header("Starting Hybrid Search Ingestion Process")
    log_info("Source: LangChain llms.txt documentation index")

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
                f"Cache file {CACHE_FILE} is corrupted. Re-extracting...")
            crawled_data = None
        except Exception as e:
            log_error(f"Error reading cache file: {e}. Re-extracting...")
            crawled_data = None

    # --- Fetch URLs from llms.txt and extract content ---
    if crawled_data is None:
        # Step 1: Get URL list from llms.txt
        url_list = await fetch_llms_txt()
        
        if not url_list:
            log_error("No URLs found in llms.txt. Exiting.")
            return
        
        log_header("Extracting Content from Documentation Pages")
        log_info(f"Processing {len(url_list)} URLs...")
        log_info("Note: Tavily Extract API has limits. Free tier: ~1000 pages/month.")
        
        # Step 2: Extract content using Tavily (one URL at a time for reliability)
        urls_to_extract = [item["url"] for item in url_list]
        extracted_results = extract_content_single(
            urls_to_extract, 
            delay=0.5  # 0.5 second between requests
        )
        
        log_success(f"Successfully extracted content from {len(extracted_results)} pages.")
        
        # Format as crawled_data structure for compatibility
        crawled_data = {"results": extracted_results}

        # Save to cache
        log_info(f"Saving extracted data to {CACHE_FILE}...")
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(crawled_data, f, indent=2)
            log_success("Successfully saved data to cache.")
        except Exception as e:
            log_error(f"Error saving data to cache: {e}")

    # --- Process extracted content ---
    valid_crawled_data = [
        page for page in crawled_data["results"]
        if page.get("raw_content")
    ]

    print(
        f"✅ Found {len(valid_crawled_data)} pages with content out of {len(crawled_data['results'])} extracted.")
    if len(valid_crawled_data) < len(crawled_data["results"]):
        print("⚠️ Warning: Some pages were skipped due to missing content.")

    if not valid_crawled_data:
        log_error("No valid content found. Exiting.")
        return

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

    # --- BM25 Sparse Encoder Training ---
    log_header("Training BM25 Sparse Encoder")
    corpus = [doc.page_content for doc in split_docs]

    # Check if BM25 params exist (for consistency across runs)
    if os.path.exists(BM25_PARAMS_FILE):
        log_info(
            f"Loading existing BM25 parameters from {BM25_PARAMS_FILE}...")
        bm25_encoder.load(BM25_PARAMS_FILE)
        log_success("BM25 encoder loaded from saved parameters.")
    else:
        log_info(f"Fitting BM25 encoder on {len(corpus)} documents...")
        bm25_encoder.fit(corpus)
        # Save BM25 params for future use
        bm25_encoder.dump(BM25_PARAMS_FILE)
        log_success(f"BM25 encoder fitted and saved to {BM25_PARAMS_FILE}")

    # --- Generate Dense Embeddings ---
    log_header("Generating Dense Embeddings")
    log_info(f"Generating embeddings for {len(split_docs)} chunks...")
    dense_embeddings = embeddings.embed_documents(corpus)
    log_success(
        f"Generated {len(dense_embeddings)} dense embeddings (dimension: {len(dense_embeddings[0])})")

    # --- Hybrid Upsert to Pinecone ---
    log_header("Storing Hybrid Vectors in Pinecone")
    hybrid_upsert_to_pinecone(
        documents=split_docs,
        dense_embeddings=dense_embeddings,
        bm25_encoder=bm25_encoder,
        batch_size=100
    )

    log_header("Hybrid Ingestion Process Completed")
    log_info(f"Total chunks stored: {len(split_docs)}")
    log_info(f"Each chunk has both dense (semantic) and sparse (BM25) vectors")
    log_info(
        f"URLs mapped: {list(set([doc.metadata['source'] for doc in split_docs]))}"
    )


if __name__ == "__main__":
    asyncio.run(main())
