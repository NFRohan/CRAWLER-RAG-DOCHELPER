import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere.rerank import CohereRerank

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# --- Configuration ---
INDEX_NAME = "doc-helper-index"
BM25_PARAMS_FILE = "bm25_params.json"

# --- 1. Setup Dense Embeddings ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2", 
    model_kwargs={"device": "cpu"}
)

# --- 2. Setup Sparse Encoder (BM25) ---
bm25_encoder = BM25Encoder()
if os.path.exists(BM25_PARAMS_FILE):
    bm25_encoder.load(BM25_PARAMS_FILE)
else:
    raise FileNotFoundError(
        f"BM25 parameters file '{BM25_PARAMS_FILE}' not found. "
        "Please run ingestion.py first to train the BM25 encoder."
    )

# --- 3. Setup Pinecone Client ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# --- 4. Hybrid Search Function ---
def hybrid_search(
    query: str, 
    top_k: int = 20, 
    alpha: float = 0.5
) -> List[Document]:
    """
    Perform hybrid search combining dense (semantic) and sparse (BM25) vectors.
    
    Args:
        query: The search query string
        top_k: Number of results to return
        alpha: Weight for dense vs sparse (0=sparse only, 1=dense only, 0.5=balanced)
    
    Returns:
        List of LangChain Document objects with search results
    """
    # Generate dense embedding for query
    dense_query = embeddings.embed_query(query)
    
    # Generate sparse vector for query using BM25
    sparse_query_dict = bm25_encoder.encode_queries([query])[0]
    sparse_query = {
        "indices": sparse_query_dict["indices"],
        "values": sparse_query_dict["values"]
    }
    
    # Scale vectors based on alpha (hybrid weighting)
    # alpha = 1: pure dense/semantic search
    # alpha = 0: pure sparse/keyword search
    # alpha = 0.5: balanced hybrid
    scaled_dense = [v * alpha for v in dense_query]
    scaled_sparse = {
        "indices": sparse_query["indices"],
        "values": [v * (1 - alpha) for v in sparse_query["values"]]
    }
    
    # Perform hybrid query
    results = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True
    )
    
    # Convert to LangChain Documents
    documents = []
    for match in results.matches:
        doc = Document(
            page_content=match.metadata.get("text", ""),
            metadata={
                "source": match.metadata.get("source", "unknown"),
                "score": match.score
            }
        )
        documents.append(doc)
    
    return documents


def hybrid_retriever(query: str) -> List[Document]:
    """Wrapper function for hybrid search to use in LangChain chain."""
    return hybrid_search(query, top_k=20, alpha=0.5)


# --- 5. Setup Reranker ---
reranker = CohereRerank(model="rerank-english-v3.0")

def rerank_docs(input_dict):
    reranked_docs = reranker.compress_documents(
        documents=input_dict["context_docs"],
        query=input_dict["question"],
    )
    # The reranker returns a list of Document objects with a 'relevance_score' in metadata.
    input_dict["context_docs"] = reranked_docs
    return input_dict

# --- 6. Setup LLM ---
llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.3,
)

# --- 7. Setup Prompt ---
template = """
You are a helpful assistant for the LangChain documentation.
Answer the question based *only* on the following context.
If you don't know the answer, just say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 8. Helper Function ---
def format_docs(docs):
    return "\n\n".join(
        f"--- Document {i+1} (Source: {doc.metadata.get('source', 'unknown')}) ---\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

# ===================================================================
# === HYBRID SEARCH RAG CHAIN ===
# Uses Sparse-Dense hybrid retrieval for better keyword + semantic search
# ===================================================================

# This is the "input" step for our chain.
# It runs the hybrid retriever and passthrough in parallel.
rag_chain_input = {
    "context_docs": RunnableLambda(hybrid_retriever),
    "question": RunnablePassthrough()
}

# This is the full chain with hybrid search and reranking.
rag_chain = (
    rag_chain_input
    # Rerank the documents based on the question
    | RunnableLambda(rerank_docs)
    # We use .assign() to add the "answer" key to the dictionary.
    # The original "context_docs" and "question" keys are passed through.
    | RunnablePassthrough.assign(
        answer=(
            # This lambda formats the input for the prompt
            (lambda x: {"context": format_docs(x["context_docs"]),
                       "question": x["question"]
                       })
            | prompt
            | llm
            | StrOutputParser()
        )
    )
)

# --- 9. Run the Chain ---
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 HYBRID SEARCH RAG SYSTEM")
    print("   Using Sparse-Dense vectors for keyword + semantic search")
    print("=" * 60)
    
    question = "What is the simplest way to get started with LangChain?"

    print(f"\nQuerying RAG chain with: '{question}'")

    # We are using .invoke() - NOT .stream()
    result = rag_chain.invoke(question)

    print("\n--- Answer ---")
    print(result["answer"])

    print("\n--- Sources ---")
    source_urls = set(doc.metadata.get('source', 'unknown')
                      for doc in result["context_docs"])
    for url in source_urls:
        print(url)

    # --- Example 2 ---
    question_2 = "What is langchain?"
    print(f"\n{'=' * 60}")
    print(f"Querying RAG chain with: '{question_2}'")

    result_2 = rag_chain.invoke(question_2)

    print("\n--- Answer ---")
    print(result_2["answer"])

    print("\n--- Sources ---")
    source_urls_2 = set(doc.metadata.get('source', 'unknown')
                          for doc in result_2["context_docs"])
    for url in source_urls_2:
        print(url)