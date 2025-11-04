import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# --- 1. Setup Embeddings ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={"device": "cpu"})

# --- 2. Setup Vector Store & Retriever ---
vector_store = PineconeVectorStore(
    index_name="doc-helper-index",
    embedding=embeddings,
)
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 5})

# --- 3. Setup LLM ---
llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.2,
)

# --- 4. Setup Prompt ---
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

# --- 5. Helper Function ---
def format_docs(docs):
    return "\n\n".join(
        f"--- Document {i+1} (Source: {doc.metadata.get('source', 'unknown')}) ---\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

# ===================================================================
# === THIS IS THE 'KNOWN GOOD' RAG CHAIN DEFINITION ===
# =Details:
# 1. `rag_chain_input` is a dict that runs two things in parallel.
# 2. It pipes its output (a dict) to `RunnablePassthrough.assign()`.
# 3. `.assign()` passes through the original dict keys AND adds a new 'answer' key.
# 4. The 'answer' key is created by running its own sub-chain.
# ===================================================================

# This is the "input" step for our chain.
# It runs the retriever and passthrough in parallel.
rag_chain_input = {
    "context_docs": retriever,  # We pass the raw documents
    "question": RunnablePassthrough()
}

# This is the full chain.
rag_chain = (
    rag_chain_input
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

# --- 6. Run the Chain ---
if __name__ == "__main__":
    question = "What is the simplest way to get started with LangChain?"

    print(f"Querying RAG chain with: '{question}'")

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
    print(f"\nQuerying RAG chain with: '{question_2}'")

    result_2 = rag_chain.invoke(question_2)

    print("\n--- Answer ---")
    print(result_2["answer"])

    print("\n--- Sources ---")
    source_urls_2 = set(doc.metadata.get('source', 'unknown')
                          for doc in result_2["context_docs"])
    for url in source_urls_2:
        print(url)