import asyncio
import os
import ssl
from typing import Any, Dict, List, Optional
import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import log_error, log_info, log_warning, log_success, log_header

load_dotenv()

#ssl with certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

embeddings = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m", model_kwargs={"device": "cpu"})
vectorstore = PineconeVectorStore(
    index_name="doc-helper-index",
    embedding=embeddings
)
tavily_extract = TavilyExtract()
travily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
travily_crawl = TavilyCrawl()







async def main ():
    """Main function to run the ingestion process."""
    log_header("Starting Ingestion Process")
    log_info("Crawling website...")

    #start crawling from a seed URL
    seed_url = "https://docs.langchain.com/oss/python/langchain/overview"
    crawled_data = travily_crawl.invoke({
        "url": seed_url,
        "max_depth": 2,
        "extract_depth" : "advanced",
    })

    all_docs = crawled_data["results"]

    log_success(f"Crawled {len(crawled_data['results'])} pages.")



if __name__ == "__main__":
    asyncio.run(main())