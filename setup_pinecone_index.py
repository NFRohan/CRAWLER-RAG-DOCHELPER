"""
Pinecone Index Setup Script for Hybrid Search (Sparse-Dense)

This script creates a Pinecone index configured for hybrid search with:
- Dense vectors: 384 dimensions (for local all-MiniLM-L12-v2 embeddings)
- Sparse vectors: BM25 keyword weights
- Metric: dotproduct (required for hybrid search)

We use LOCAL embeddings (HuggingFace sentence-transformers) instead of 
Pinecone's integrated embedding models for full control and offline capability.

Run this script once before ingestion to set up the index.

Usage:
    python setup_pinecone_index.py          # Create index
    python setup_pinecone_index.py create   # Create index
    python setup_pinecone_index.py delete   # Delete index
    python setup_pinecone_index.py stats    # Show index statistics
"""

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from logger import log_error, log_info, log_success, log_warning, log_header

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
INDEX_NAME = os.getenv("PINECONE_INDEX", "doc-helper-index")
DIMENSION = 384  # Dimension for sentence-transformers/all-MiniLM-L12-v2 (LOCAL embedding)
METRIC = "dotproduct"  # Required for hybrid search with sparse vectors
CLOUD = "aws"
REGION = os.getenv("PINECONE_REGION", "us-east-1")


def get_pinecone_client() -> Pinecone:
    """Initialize and return a Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        log_error("PINECONE_API_KEY not found in .env file")
        raise ValueError("PINECONE_API_KEY not found in .env file")
    return Pinecone(api_key=api_key)


def create_hybrid_index():
    """
    Create a Pinecone index configured for hybrid (sparse-dense) search.
    
    This uses LOCAL embeddings (HuggingFace all-MiniLM-L12-v2) rather than
    Pinecone's integrated embedding models, giving you:
    - Full control over the embedding process
    - Offline embedding capability
    - Hybrid search with BM25 sparse vectors
    """
    log_header("Pinecone Hybrid Index Setup (Local Embeddings)")
    
    pc = get_pinecone_client()
    
    # Check if index already exists using the new API pattern
    if pc.has_index(INDEX_NAME):
        log_warning(f"Index '{INDEX_NAME}' already exists.")
        
        # Get index info
        index_info = pc.describe_index(INDEX_NAME)
        log_info("Current index configuration:")
        log_info(f"  - Dimension: {index_info.dimension}")
        log_info(f"  - Metric: {index_info.metric}")
        log_info(f"  - Host: {index_info.host}")
        
        # Check if it's configured correctly for hybrid search
        if index_info.metric != METRIC:
            log_warning(f"Index metric is '{index_info.metric}', but hybrid search requires '{METRIC}'.")
            log_warning("You may need to delete and recreate the index for hybrid search to work properly.")
            
            user_input = input("\nDo you want to delete and recreate the index? (yes/no): ").strip().lower()
            if user_input == "yes":
                log_info(f"Deleting index '{INDEX_NAME}'...")
                pc.delete_index(INDEX_NAME)
                log_success(f"Index '{INDEX_NAME}' deleted.")
                # Wait for deletion to propagate
                log_info("Waiting for deletion to propagate...")
                time.sleep(3)
            else:
                log_info("Keeping existing index. Hybrid search may not work correctly.")
                return
        else:
            log_success("Index is already configured for hybrid search!")
            return
    
    # Create new index with hybrid search configuration
    # NOTE: We use create_index() (not create_index_for_model()) because we use LOCAL embeddings
    log_info(f"Creating new index '{INDEX_NAME}' with hybrid search configuration...")
    log_info(f"  - Dimension: {DIMENSION} (for local all-MiniLM-L12-v2 embeddings)")
    log_info(f"  - Metric: {METRIC} (required for sparse-dense hybrid)")
    log_info(f"  - Cloud: {CLOUD}")
    log_info(f"  - Region: {REGION}")
    log_info("  - Embedding: LOCAL (HuggingFace sentence-transformers)")
    
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
    )
    
    # Wait for index to be ready
    log_info("Waiting for index to be ready...")
    while not pc.describe_index(INDEX_NAME).status.get("ready", False):
        time.sleep(1)
        print(".", end="", flush=True)
    print()
    
    log_success(f"Index '{INDEX_NAME}' created successfully!")
    
    # Display final index info
    index_info = pc.describe_index(INDEX_NAME)
    log_info(f"Index host: {index_info.host}")
    log_header("Setup Complete")
    log_info("You can now run ingestion.py to populate the index with documents.")
    log_info("Documents will be embedded locally using HuggingFace sentence-transformers.")


def delete_index():
    """Delete the Pinecone index (utility function)."""
    pc = get_pinecone_client()
    
    if not pc.has_index(INDEX_NAME):
        log_warning(f"Index '{INDEX_NAME}' does not exist.")
        return
    
    log_info(f"Deleting index '{INDEX_NAME}'...")
    pc.delete_index(INDEX_NAME)
    log_success(f"Index '{INDEX_NAME}' deleted successfully!")


def get_index_stats():
    """Get statistics about the current index."""
    pc = get_pinecone_client()
    
    if not pc.has_index(INDEX_NAME):
        log_warning(f"Index '{INDEX_NAME}' does not exist.")
        return
    
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    
    log_header("Index Statistics")
    log_info(f"Index name: {INDEX_NAME}")
    log_info(f"Total vectors: {stats.get('total_vector_count', 0)}")
    log_info(f"Dimension: {stats.get('dimension', DIMENSION)}")
    log_info(f"Embedding: LOCAL (HuggingFace all-MiniLM-L12-v2)")
    
    namespaces = stats.get('namespaces', {})
    if namespaces:
        log_info("Namespaces:")
        for ns_name, ns_info in namespaces.items():
            log_info(f"  - {ns_name or '(default)'}: {ns_info.get('vector_count', 0)} vectors")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "delete":
            delete_index()
        elif command == "stats":
            get_index_stats()
        elif command == "create":
            create_hybrid_index()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python setup_pinecone_index.py [create|delete|stats]")
    else:
        create_hybrid_index()
